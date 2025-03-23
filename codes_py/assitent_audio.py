import speech_recognition as sr
import os
import re
from playsound import playsound
from gtts import gTTS
from pydub import AudioSegment
import threading
import queue
import time
import logging
import pyaudio
import numpy as np
import wave

import subprocess
import ollama

# Configurar logging para ayudar a depurar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("asistente_voz.log"), logging.StreamHandler()]
)
logger = logging.getLogger("AsistenteVoz")

class ActivacionVoz:
    def __init__(self, command_queue):
        # Crear el reconocedor de voz
        self.recognizer = sr.Recognizer()
        
        # Ajustar parámetros de reconocimiento para mejor sensibilidad
        self.recognizer.energy_threshold = 300  # Ajustar según el entorno (más bajo = más sensible)
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.6  # Reducido para detectar pausas más cortas
        self.recognizer.operation_timeout = None  # Sin timeout para operaciones API
        
        # Palabras clave de activación (múltiples opciones para mayor probabilidad de detección)
        self.palabras_activacion = ["silvia", "asistente", "ayuda", "hey", "hola", "escucha"]
        
        # Inicializar el micrófono con selección de dispositivo
        self.seleccionar_microfono()
        
        # Cola para enviar comandos al hilo principal
        self.command_queue = command_queue
        
        # Variable para controlar el ciclo de comandos
        self.escuchando = True
        
        # Semáforo para evitar que el sistema se escuche a sí mismo
        self.reproduciendo_audio = threading.Event()
        
        # Contador de detecciones fallidas para ajuste dinámico
        self.fallos_consecutivos = 0
        
        # Última vez que se ajustó el ruido ambiental
        self.ultimo_ajuste = 0
        
        # Indicador de actividad por audio
        self.audio_callback_activo = False
        
        # Crear directorio de caché si no existe
        self.cache_dir = "audio_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def seleccionar_microfono(self):
        """Selecciona siempre el micrófono 0"""
        try:
            self.mic = sr.Microphone(device_index=0)
            logger.info("Micrófono seleccionado: 0")
        except Exception as e:
            logger.error(f"Error al seleccionar micrófono: {e}")
            self.mic = sr.Microphone()
            logger.info("Usando micrófono predeterminado")

    def detectar_nivel_audio(self, duracion=5):
        """Analiza el nivel de audio ambiental para calibrar mejor los parámetros"""
        p = pyaudio.PyAudio()
        
        # Configuración para análisis de audio
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        logger.info(f"Analizando nivel de audio ambiental durante {duracion} segundos...")
        
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            
            frames = []
            niveles = []
            
            # Recoger datos durante 'duracion' segundos
            for i in range(0, int(RATE / CHUNK * duracion)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Convertir a numpy array para análisis
                audio_data = np.frombuffer(data, dtype=np.int16)
                nivel = np.abs(audio_data).mean()
                niveles.append(nivel)
            
            # Cerrar stream
            stream.stop_stream()
            stream.close()
            
            # Analizar niveles
            nivel_promedio = np.mean(niveles)
            nivel_maximo = np.max(niveles)
            nivel_minimo = np.min(niveles)
            
            logger.info(f"Nivel promedio: {nivel_promedio}, Máximo: {nivel_maximo}, Mínimo: {nivel_minimo}")
            
            # Ajustar energy_threshold basado en el análisis
            if nivel_promedio > 0:
                # Establecer un nivel por encima del ruido ambiental pero no demasiado alto
                nuevo_threshold = nivel_promedio * 1.5
                
                # Asegurar que esté en un rango razonable
                nuevo_threshold = max(200, min(4000, nuevo_threshold))
                
                # Actualizar reconocedor
                self.recognizer.energy_threshold = nuevo_threshold
                logger.info(f"Energy threshold ajustado a: {nuevo_threshold}")
                
                # Guardar muestra de audio para depuración si necesario
                wf = wave.open(os.path.join(self.cache_dir, "ambiente_sample.wav"), 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                return nuevo_threshold
                
        except Exception as e:
            logger.error(f"Error al analizar nivel de audio: {e}")
        finally:
            p.terminate()
        
        return 300  # Valor por defecto si falló el análisis

    def reproducir_audio(self, mensaje, velocidad=1.2):
        try:
            # Marcar que estamos reproduciendo audio para evitar escuchar nuestra propia salida
            self.reproduciendo_audio.set()
            
            # Nombre de archivo único basado en hash del mensaje para caché
            import hashlib
            mensaje_hash = hashlib.md5(mensaje.encode()).hexdigest()
            archivo_audio = os.path.join(self.cache_dir, f"respuesta_{mensaje_hash}.mp3")
            
            # Verificar si ya tenemos este mensaje en caché
            if not os.path.exists(archivo_audio):
                # Generar audio con gTTS
                tts = gTTS(mensaje, lang='es', slow=False)
                tts.save(archivo_audio)
                
                # Ajustar velocidad
                audio = AudioSegment.from_file(archivo_audio)
                audio_rapido = audio.speedup(playback_speed=velocidad)
                audio_rapido.export(archivo_audio, format="mp3")
            
            # Reproducir audio (con indicador visual opcional)
            logger.info(f"Reproduciendo: {mensaje}")
            print(f"\n🔊 Asistente: {mensaje}")
            
            # Reproducir un sonido corto antes para alertar
            playsound(archivo_audio)
            
            # Esperar un momento después de reproducir para evitar capturar el eco
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error al reproducir audio: {e}")
            print(f"Error al reproducir: {e}")
        finally:
            # Desmarcar la reproducción de audio
            self.reproduciendo_audio.clear()

    def escuchar(self, tiempo_espera=7, limite_frase=5):
        # Si estamos reproduciendo audio, esperar hasta que termine
        if self.reproduciendo_audio.is_set():
            logger.info("Esperando a que termine la reproducción de audio...")
            self.reproduciendo_audio.wait()
            time.sleep(0.5)  # Pequeña pausa adicional
        
        # Crear un indicador visual de escucha
        threading.Thread(target=self._mostrar_indicador_escucha, daemon=True).start()
        
        with self.mic as source:
            logger.info(f"Escuchando... (timeout={tiempo_espera}s, phrase_limit={limite_frase}s)")
            print("\n🎤 Escuchando...", end="", flush=True)
            
            try:
                # Ajustamos por ruido ambiental periódicamente
                tiempo_actual = time.time()
                if tiempo_actual - self.ultimo_ajuste > 60:  # Una vez por minuto
                    logger.info("Ajustando para ruido ambiental...")
                    try:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                        self.ultimo_ajuste = tiempo_actual
                    except Exception as e:
                        logger.error(f"Error ajustando ruido ambiental: {e}")
                
                # Listen con parámetros ajustados
                audio = self.recognizer.listen(
                    source, 
                    timeout=tiempo_espera, 
                    phrase_time_limit=limite_frase
                )
                
                logger.info("Audio capturado correctamente")
                print(" ✓", flush=True)
                
                # Guardar el audio para depuración si es necesario
                timestamp = int(time.time())
                with open(os.path.join(self.cache_dir, f"audio_{timestamp}.wav"), "wb") as f:
                    f.write(audio.get_wav_data())
                
                return audio
            except sr.WaitTimeoutError:
                logger.warning("Tiempo de espera agotado. No se detectó audio.")
                print(" ⚠️ (timeout)", flush=True)
                return None
            except Exception as e:
                logger.error(f"Error al escuchar: {e}")
                print(f" ❌ (error: {str(e)})", flush=True)
                return None
            finally:
                self.audio_callback_activo = False

    def _mostrar_indicador_escucha(self):
        """Muestra un indicador animado mientras está escuchando"""
        self.audio_callback_activo = True
        indicadores = ['.', '..', '...', '....']
        i = 0
        while self.audio_callback_activo:
            print(f"\r🎤 Escuchando{indicadores[i]}", end="", flush=True)
            i = (i + 1) % len(indicadores)
            time.sleep(0.3)

    def transcribir_audio(self, audio):
        if audio is None:
            return ""
        
        # Intentar múltiples servicios de reconocimiento si es necesario
        servicios = [
            (self._transcribir_google, "Google"),
        ]
        
        for servicio_func, nombre_servicio in servicios:
            try:
                texto = servicio_func(audio)
                if texto and len(texto.strip()) > 0:
                    logger.info(f"Transcripción ({nombre_servicio}): '{texto}'")
                    return texto.lower()  # Convertir a minúsculas para mejor comparación
            except Exception as e:
                logger.warning(f"Error con servicio {nombre_servicio}: {e}")
                continue
        
        logger.warning("Todos los servicios de transcripción fallaron")
        return ""

    def _transcribir_google(self, audio):
        try:
            return self.recognizer.recognize_google(audio, language="es-ES")
        except sr.UnknownValueError:
            logger.warning("Google no pudo entender el audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Error al solicitar el servicio de Google: {e}")
            raise


    def detectar_palabra_clave(self, audio):
        texto = self.transcribir_audio(audio)
        
        if not texto:
            return False
            
        logger.info(f"Verificando si '{texto}' contiene palabra clave")
        
        # Método mejorado: buscar coincidencias parciales o similares
        for palabra in self.palabras_activacion:
            if palabra in texto:
                logger.info(f"¡Palabra de activación detectada! ('{palabra}' en '{texto}')")
                self.reproducir_audio("¿En qué puedo ayudarte?")
                return True
                
        # Método alternativo: verificar similitud de palabras
        texto_palabras = texto.split()
        for palabra_entrada in texto_palabras:
            for palabra_clave in self.palabras_activacion:
                # Calcular similitud simple (puedes reemplazar con Levenshtein si es necesario)
                if self._palabras_similares(palabra_entrada, palabra_clave, umbral=0.7):
                    logger.info(f"¡Palabra similar detectada! ('{palabra_entrada}' similar a '{palabra_clave}')")
                    self.reproducir_audio("¿En qué puedo ayudarte?")
                    return True
                
        return False

    def _palabras_similares(self, palabra1, palabra2, umbral=0.7):
        """Compara similitud entre palabras usando similitud de caracteres compartidos"""
        # Esta es una implementación simple, puedes usar Levenshtein para mayor precisión
        if len(palabra1) < 3 or len(palabra2) < 3:
            return False
            
        palabra1 = palabra1.lower()
        palabra2 = palabra2.lower()
        
        # Si una es prefijo de la otra
        if palabra1.startswith(palabra2) or palabra2.startswith(palabra1):
            return True
            
        # Calcular caracteres compartidos
        caracteres_comunes = sum(1 for c in palabra1 if c in palabra2)
        longitud_max = max(len(palabra1), len(palabra2))
        similitud = caracteres_comunes / longitud_max
        
        return similitud >= umbral

    def command_ollama(self, text):
        """
        Llama a Ollama utilizando el modelo llama3.2:1b en modo streaming.
        Se define un contexto en el que el robot es un UR5e dibujante de arte llamado SILVIA.
        """
        messages = [
            {
                "role": "system", 
                "content": ("""
                    Eres un robot UR5e dibujante de arte llamado SILVIA y tienes respuestas cortas y concisas.
                    Tu nombre es SILVIA 
                    Eres muy bueno dibujando y te gusta interactuar con las personas, eres un gran conocedor sobre el arte y
                    puedes reposnder cualquier pregunta sobre ello, puedes dar ideas de inspiración para crear obras de arte de manera simple.
                    Puedes recomendar libros, películas y artistas para inspirar a las personas.
                    Puedes dibujar cualquier cosa que te pidan, desde retratos hasta paisajes.
                    
                            
                    Da respuestas simples y concisas, y evita respuestas largas y detalladas. no uses simolos de puntuación ni *
                    No hagas listas simpre da una opción y no más.
                    Siempre responde de manera positiva y amigable.
                    No hagas preguntas abiertas, siempre da una respuesta.
                    No hagas preguntas personales.
                    No hagas ningun tipo de pregunta.
                    Simpre debes de recomendar algo sobre arte.
                    Enfocate en estilos aquitectonicos."""
                )
            },
            {"role": "user", "content": "Hola, ¿cómo estás?"},
            {"role": "system", "content": "Hola, estoy bien, ¿en qué puedo ayudarte?"},
            {"role": "user", "content": "¿Cuál es tu nombre?"},
            {"role": "system", "content": "Mi nombre es SILVIA"},
            {"role": "user", "content": "¿Qué puedes hacer?"},
            {"role": "system", "content": "Soy un robot UR5e dibujante de arte llamado SILVIA. Puedo responder preguntas sobre arte, dar ideas de inspiración para crear obras de arte y dibujar cualquier cosa que me pidas."},
            {
                "role": "user",
                "content": text
            }
        ]
        
        # Inicia el chat en modo streaming con el modelo especificado
        stream = ollama.chat(model='llama3.2:1b', messages=messages, stream=True)
        
        full_response = ""
        for chunk in stream:
            contenido = chunk['message']['content']
            print(contenido, end='', flush=True)
            full_response += contenido
        print('')  # Salto de línea final
        return full_response

    def command_stable_diffusion(self):
        """
        Función para generar un prompt para Stable Diffusion.
        1. Pregunta al usuario qué desea dibujar.
        2. Utiliza Ollama para generar un prompt en inglés basado en la descripción.
        El prompt se configura para que sea simple, fácil de dibujar, con estilo artístico y animado.
        """
        # 1. Preguntar al usuario qué desea dibujar
        self.reproducir_audio("¿Qué deseas dibujar?")
        audio_dibujo = self.escuchar(10, 7)  # Escucha durante 10 segundos con límite de frase de 7 segundos
        if audio_dibujo is None:
            self.reproducir_audio("No se entendió lo que deseas dibujar. Inténtalo de nuevo.")
            return "No se entendió lo que deseas dibujar.", None
        dibujo_text = self.transcribir_audio(audio_dibujo)
        if not dibujo_text:
            self.reproducir_audio("No se entendió lo que deseas dibujar. Inténtalo de nuevo.")
            return "No se entendió lo que deseas dibujar.", None

        # 2. Generar prompt para Stable Diffusion con Ollama
        prompt_system = (
            "You are an assistant that generates prompts for Stable Diffusion. "
            "Generate a prompt in English based on the following description. "
            "Make it simple, easy to draw, artistic, and animated in style."
        )
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": dibujo_text}
        ]
        stream = ollama.chat(model='llama3.2:1b', messages=messages, stream=True)
        sd_prompt = ""
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            sd_prompt += content
        print('')  # Salto de línea final

        self.reproducir_audio("Prompt generado para Stable Diffusion.")
        print(f"Prompt generado: {sd_prompt}")
        return f"Prompt generado: {sd_prompt}", None


    def procesar_comando(self, texto):
        """
        Procesa el comando de texto y devuelve una respuesta y un código de comando.
        Si el comando no coincide con ninguno de los predefinidos, se utiliza Ollama para generar la respuesta.
        Además, si se detectan las palabras 'dibujar', 'pintar' o 'crear', se invoca la funcionalidad para
        generar un prompt para Stable Diffusion.
        """
        if not texto or len(texto.strip()) < 2:
            return "No he podido entender tu instrucción. ¿Podrías repetirla?", None

        # Comandos predefinidos
        comandos = {
            "ayuda":   (0, "Puedo ayudarte con varios comandos básicos como: hora, fecha, saludo, etc."),
            "hora":    (1, f"Son las {time.strftime('%H:%M')}"),
            "fecha":   (2, f"Hoy es {time.strftime('%d de %B de %Y')}"),
            "saludo":  (3, "Hola, ¿cómo estás?"),
            "nombre":  (4, "Mi nombre es Asistente de Voz"),
            "gracias": (5, "De nada, estoy para ayudarte"),
            "adios":   (6, "Hasta luego, que tengas un buen día"),
            "salir":   (7, "Cerrando asistente")
        }

        # Buscar coincidencias con comandos conocidos
        for palabra_clave, (codigo, respuesta) in comandos.items():
            if palabra_clave in texto:
                return respuesta, codigo

        # Si se detectan palabras clave para dibujar, pintar o crear, ejecutar la funcionalidad de prompt
        dibujar_keywords = ["dibuja", "pinta", "crea"]
        if any(kw in texto for kw in dibujar_keywords):
            return self.command_stable_diffusion()

        # Si no se detecta un comando específico, se utiliza Ollama para generar una respuesta general
        respuesta_ollama = self.command_ollama(texto)
        return respuesta_ollama, None


    def ciclo_de_comandos(self):
        self.fallos_consecutivos = 0
        
        # Realizar análisis inicial de nivel de audio
        self.detectar_nivel_audio(duracion=3)
        
        # Anunciar que el asistente está listo
        self.reproducir_audio("Asistente de voz activado. Di mi nombre para comenzar.")
        
        while self.escuchando:
            try:
                logger.info("Esperando la palabra clave...")
                
                # Ajustar parámetros de escucha según el número de fallos consecutivos
                tiempo_espera = 7 + min(self.fallos_consecutivos * 0.5, 5)  # Máx +5 segundos
                limite_frase = 5 + min(self.fallos_consecutivos * 0.5, 3)   # Máx +3 segundos
                
                # Si hay muchos fallos, recalibrar
                if self.fallos_consecutivos > 10:
                    logger.warning(f"Muchos fallos consecutivos ({self.fallos_consecutivos}), recalibrando...")
                    nuevo_threshold = self.detectar_nivel_audio(duracion=3)
                    self.reproducir_audio(f"Recalibrando sensibilidad a {int(nuevo_threshold)}")
                    self.fallos_consecutivos = 0
                
                # Escuchar palabra clave
                audio = self.escuchar(tiempo_espera, limite_frase)
                
                if audio is None:
                    self.fallos_consecutivos += 1
                    continue
                
                # Detectar palabra clave
                if self.detectar_palabra_clave(audio):
                    self.fallos_consecutivos = 0  # Reiniciar contador de fallos
                    logger.info("Palabra clave detectada. Entrando en modo de instrucciones.")
                    
                    # Bucle interno para recibir comandos después de activación
                    intentos_comando = 0
                    while intentos_comando < 3:  # Máximo 3 intentos por activación
                        logger.info("Esperando instrucciones...")
                        audio_comando = self.escuchar(10, 7)  # Tiempo más largo para comandos
                        
                        if audio_comando is None:
                            intentos_comando += 1
                            self.reproducir_audio("¿Puedes repetir tu instrucción, por favor?")
                            continue
                            
                        texto = self.transcribir_audio(audio_comando)
                        
                        if not texto:
                            intentos_comando += 1
                            self.reproducir_audio("No te he entendido. ¿Puedes repetir?")
                            continue
                            
                        logger.info(f"Instrucción recibida: {texto}")
                        respuesta, comando = self.procesar_comando(texto)
                        
                        # Enviar el comando a la cola si existe
                        if comando is not None:
                            self.command_queue.put(comando)
                            
                        # Reproducir respuesta
                        self.reproducir_audio(respuesta)
                        
                        # Si el comando es salir, terminar
                        if comando == 7:  # código para salir
                            self.escuchando = False
                        
                        # Pequeña pausa antes de volver a escuchar la palabra clave
                        time.sleep(1)
                        break
                    
                    if intentos_comando >= 3:
                        logger.info("Máximo de intentos alcanzado. Volviendo a esperar palabra clave.")
                        self.reproducir_audio("Volviendo a modo de espera. Di mi nombre para activarme de nuevo.")
                else:
                    # Si no detectamos palabra clave, incrementamos contador de fallos
                    self.fallos_consecutivos += 1
                
            except Exception as e:
                logger.error(f"Error en el ciclo de comandos: {e}")
                self.fallos_consecutivos += 1
                time.sleep(1)  # Pausa para evitar bucles rápidos en caso de error

    def iniciar_hilo(self):
        logger.info("Iniciando hilo de reconocimiento de voz...")
        self.hilo = threading.Thread(target=self.ciclo_de_comandos)
        self.hilo.daemon = True
        self.hilo.start()

    def detener(self):
        logger.info("Deteniendo el asistente de voz...")
        self.escuchando = False
        if hasattr(self, 'hilo') and self.hilo.is_alive():
            self.hilo.join(timeout=5)
            if self.hilo.is_alive():
                logger.warning("El hilo no se cerró correctamente. Continuando...")
            else:
                logger.info("Hilo terminado correctamente.")
        else:
            logger.info("El hilo de escucha ya ha terminado o no se inició.")
        
        # Limpiar archivos de caché opcionales
        try:
            import shutil
            respuesta = input("¿Quieres eliminar la caché de audio? (s/n): ")
            if respuesta.lower() in ['s', 'si', 'sí', 'y', 'yes']:
                shutil.rmtree(self.cache_dir)
                print("Caché de audio eliminada.")
        except Exception as e:
            logger.error(f"Error al limpiar caché: {e}")

def test_mic():
    """Función para probar el micrófono antes de iniciar el asistente"""
    try:
        print("\n===== PRUEBA DE MICRÓFONO =====")
        print("Probando micrófono...")
        
        # Analizar nivel de audio ambiente
        p = pyaudio.PyAudio()
        
        # Obtener información del dispositivo 0 para usar su tasa de muestreo por defecto
        device_info = p.get_device_info_by_index(0)
        RATE = int(device_info["defaultSampleRate"])
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        DURACION = 3
        
        print(f"Analizando nivel de audio ambiental durante {DURACION} segundos...")
        print("Por favor, mantén silencio...")
        
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        niveles = []
        
        # Recoger datos durante 'duracion' segundos
        for i in range(0, int(RATE / CHUNK * DURACION)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convertir a numpy array para análisis
            audio_data = np.frombuffer(data, dtype=np.int16)
            nivel = np.abs(audio_data).mean()
            niveles.append(nivel)
            
            # Mostrar indicador visual
            barra = "#" * int(nivel / 100)
            print(f"\rNivel: {nivel:.1f} [{barra.ljust(40)}]", end="")
            
            time.sleep(0.01)
        
        # Cerrar stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Analizar niveles
        nivel_promedio = np.mean(niveles)
        nivel_maximo = np.max(niveles)
        nivel_minimo = np.min(niveles)
        
        print(f"\n\nNivel promedio: {nivel_promedio:.1f}, Máximo: {nivel_maximo:.1f}, Mínimo: {nivel_minimo:.1f}")
        
        # Sugerir threshold basado en el análisis
        threshold_sugerido = nivel_promedio * 1.5
        threshold_sugerido = max(200, min(4000, threshold_sugerido))
        
        print(f"Threshold sugerido: {threshold_sugerido:.1f}")
        
        # Prueba real de reconocimiento con el micrófono 0
        r = sr.Recognizer()
        r.energy_threshold = threshold_sugerido
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.6
        
        print("\nAhora di algo para comprobar el reconocimiento...")
        with sr.Microphone(device_index=0) as source:
            r.adjust_for_ambient_noise(source, duration=1.5)
            audio = r.listen(source, timeout=7, phrase_time_limit=7)
            
            print("Procesando audio...")
            try:
                texto = r.recognize_google(audio, language="es-ES")
                print(f"Prueba exitosa! Has dicho: '{texto}'")
                return True, threshold_sugerido
            except sr.UnknownValueError:
                print("No se pudo entender el audio en la prueba")
                return False, threshold_sugerido
            except sr.RequestError as e:
                print(f"Error al solicitar el servicio de reconocimiento: {e}")
                return False, threshold_sugerido
            
    except Exception as e:
        print(f"Error inesperado en la prueba del micrófono: {e}")
        return False, 300


if __name__ == '__main__':
    # Probar el micrófono primero
    mic_ok, threshold_inicial = test_mic()
    if not mic_ok:
        print("No se pudo probar el micrófono. Intentando con valores por defecto...")
    
    # Crear una cola para recibir comandos del asistente
    command_queue = queue.Queue()
    
    # Instanciar el asistente de voz
    asistente = ActivacionVoz(command_queue)
    asistente.recognizer.energy_threshold = threshold_inicial
    
    # Iniciar el hilo del asistente
    asistente.iniciar_hilo()
    
    print("\nEl asistente está activo. Presiona Ctrl+C para terminar.")
    print("Di alguna de las siguientes palabras para activar el asistente:")
    print(", ".join(asistente.palabras_activacion))
    print("\nComandos disponibles:")
    print("- ayuda: Mostrar comandos disponibles")
    print("- hora: Consultar la hora actual")
    print("- fecha: Consultar la fecha actual")
    print("- saludo: Saludar al asistente")
    print("- nombre: Preguntar el nombre del asistente")
    print("- gracias: Agradecer al asistente")
    print("- adios/salir: Finalizar el programa")
    
    try:
        # Bucle principal que monitorea la cola de comandos
        while True:
            if not command_queue.empty():
                comando = command_queue.get()
                print(f"\n📋 Comando recibido: {comando}")
                
                # Si el comando es salir, terminar
                if comando == 7:
                    print("Comando de salida recibido. Terminando programa...")
                    asistente.detener()
                    break
                
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDeteniendo el asistente...")
        asistente.detener()
        print("Programa finalizado.")