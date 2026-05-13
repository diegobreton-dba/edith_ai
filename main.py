import asyncio
import tempfile
import uuid
import pygame
import edge_tts
import os
import time
import webbrowser
import ollama
import sounddevice as sd
import queue
import json
import soundfile as sf
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
from google import genai

load_dotenv()

USER_NAME = "Diego"
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]
CURRENT_GEMINI_MODEL = None
LOCAL_MODEL = "gemma3:1b"
WAKE_WORDS = ["edith", "edit", "edid", "edif"]
VOICE_ENABLED = True
VOICE = "es-ES-ElviraNeural"
VOICE_RATE = "+5%"
EXAM_VOICE_RATE = "-8%"
VOICE_VOLUME = "+0%"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

current_topic = None
exam_mode = False
memory = []
AUDIO_QUEUE = queue.Queue()

VOSK_MODEL = Model("vosk-model-small-es-0.42")

recognizer = KaldiRecognizer(
    VOSK_MODEL,
    16000
)

def normalize(text):
    return (
        text.lower().strip()
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
        .replace("¿", "").replace("?", "").replace("¡", "").replace("!", "")
        .replace(",", "").replace(".", "")
    )

def ideal_length(text):

    words = len(text.split())

    if exam_mode:
        return 8 <= words <= 18

    return 14 <= words <= 35

async def speak_async(text):
    if not VOICE_ENABLED:
        return

    if not text or not text.strip():
        return

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"edith_voice_{uuid.uuid4().hex}.mp3"
    )

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=EXAM_VOICE_RATE if exam_mode else VOICE_RATE,
        volume=VOICE_VOLUME
    )

    await communicate.save(temp_path)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.05)

    pygame.mixer.music.unload()
    pygame.mixer.quit()

    try:
        os.remove(temp_path)
    except:
        pass

def clean_output(text):

    if not text:
        return "No pude responder, señor."

    text = text.strip()

    bad_starts = [
        "EDITH:",
        "Respuesta:",
        "Modo examen:",
        "Usuario:",
    ]

    for bad in bad_starts:

        if text.startswith(bad):
            text = text.replace(bad, "").strip()

    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("\n", " ")

    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()

def validate_output(text):

    if not text:
        return False

    words = len(text.split())

    if exam_mode:

        if words < 6:
            return False

        if words > 22:
            return False

    else:

        if words < 10:
            return False

        if words > 45:
            return False

    forbidden = [
        "modo examen",
        "¿en qué puedo ayudarte?",
        "markdown",
        "usuario:",
        "edith:"
    ]

    lowered = text.lower()

    for item in forbidden:

        if item in lowered:
            return False

    return True

def speak(text):
    try:
        asyncio.run(speak_async(text))
    except Exception as e:
        print(f"[Error de voz]: {e}")

def play_sound(path):

    try:

        pygame.mixer.init()

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    except Exception as e:

        print(f"[Sound Error]: {e}")

def audio_callback(indata, frames, time, status):

    if status:
        return

    AUDIO_QUEUE.put(bytes(indata))

def record_command(seconds=6):

    print("Escuchando comando...")

    recording = sd.rec(
        int(seconds * 16000),
        samplerate=16000,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    filename = "temp_command.wav"

    sf.write(
        filename,
        recording,
        16000
    )

    return filename

def transcribe_audio_gemini(audio_path):

    try:

        uploaded_file = client.files.upload(
            file=audio_path
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Transcribe exactamente este audio en español.",
                uploaded_file
            ]
        )

        text = normalize(
            response.text.strip()
        )

        return text

    except Exception as e:

        print(f"[STT ERROR]: {e}")

        return ""

def listen_microphone():
    waiting_for_command = False
    last_wake_time = 0
    command_timeout = 8

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        print("EDITH escuchando, señor...")
        print("Diga 'Edith' y luego su comando.")

        while True:
            data = AUDIO_QUEUE.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                raw_text = result.get("text", "")
                text = normalize(raw_text)

                if not text:
                    continue

                print(f"\nRAW VOSK: {raw_text}")
                print(f"NORMALIZADO: {text}")

                now = time.time()

                # Si está esperando comando y se acabó el tiempo
                if waiting_for_command and now - last_wake_time > command_timeout:
                    waiting_for_command = False
                    print("EDITH volvió a modo espera.")
                    continue

                words = text.split()
                first_word = words[0] if words else ""

                # Modo dormida: solo reacciona a wake word
                if not waiting_for_command:
                    if first_word in WAKE_WORDS:
                        command_after_wake = " ".join(words[1:]).strip()

                        play_sound("sounds/activate.mp3")

                        if command_after_wake:
                            waiting_for_command = False
                            full_command = f"edith {command_after_wake}"
                            process_and_speak(full_command)

                        else:
                            waiting_for_command = True
                            last_wake_time = now

                            answer = f"A su orden, {USER_NAME}."
                            print(f"EDITH: {answer}")
                            speak(answer)

                    else:
                        print("Ruido/frase ignorada. EDITH dormida.")

                # Modo despierta: procesa lo siguiente que digas sin repetir Edith
                else:

                    waiting_for_command = False

                    audio_path = record_command(6)

                    transcribed = transcribe_audio_gemini(audio_path)

                    if not transcribed:
                        print("No pude entender el comando.")
                        continue

                    print(f"\nTRANSCRIPCIÓN GEMINI: {transcribed}")

                    full_command = f"edith {transcribed}"

                    process_and_speak(full_command)

def get_command(text):
    words = normalize(text).split()
    if not words or words[0] not in WAKE_WORDS:
        return None
    return " ".join(words[1:]).strip()

def spotify(command):
    query = command.replace("reproduce", "").replace("pon", "").replace("spotify", "").strip()
    if not query:
        return "Diga la canción, señor."
    webbrowser.open(f"https://open.spotify.com/search/{query.replace(' ', '%20')}")
    return f"Buscando {query} en Spotify, señor."

def formula_answer(command):
    text = normalize(command)
    if "velocidad" in text and ("formula" in text or "ecuacion" in text):
        return "Velocidad igual distancia entre tiempo."
    if "densidad" in text and ("formula" in text or "ecuacion" in text):
        return "Densidad igual masa entre volumen."
    if "circunferencia" in text and ("formula" in text or "ecuacion" in text):
        return "Ecuación canónica: x menos h al cuadrado más y menos k al cuadrado igual radio al cuadrado."
    return None

def quick_answer(command):
    text = normalize(command)
    if text == "":
        return f"A su orden, {USER_NAME}."
    if "estado de gemini" in text:
        return f"Conectada a {CURRENT_GEMINI_MODEL or 'Gemini Cloud'}, señor."
    if "modo examen estado" in text:
        return "Modo examen activo, señor." if exam_mode else "Modo examen inactivo, señor."
    if "como estas" in text:
        return "Operativa y lista para asistirle, señor."
    if "quien eres" in text:
        return "Soy EDITH, su asistente personal integrada en gafas inteligentes."
    if "que puedes hacer" in text:
        return "Puedo responder preguntas, abrir Spotify, resolver fórmulas y asistirle con información rápida, señor."
    if (
    "quien te creo" in text
    or "quien te creó" in text
    or "quien te hizo" in text
    or "quien te desarrollo" in text
    or "quien te desarrolló" in text
    or "que te creo" in text
    or "que te creó" in text
    or "tu creador" in text
    ):
        return "Fui desarrollada por Diego Breton, mi creador y usuario autorizado principal, señor."
    if "donde estan tus servidores" in text or "donde se encuentran tus servidores" in text:
        return "Mi núcleo actual funciona entre este equipo y servicios externos cuando están disponibles, señor."
    if "quien te creo" in text or "quien te creó" in text:
        return "Fui desarrollada por Diego Breton como asistente personal experimental, señor."
    if "quien es diego breton" in text:
        return "Diego Breton es mi desarrollador principal y usuario autorizado nivel alfa."
    if "donde estan tus servidores" in text or "donde se encuentran tus servidores" in text:
        return "Parte de mi infraestructura está distribuida y parte permanece clasificada, señor."
    if "quien puede usarte" in text:
        return "Mi acceso prioritario está reservado para Diego Breton, señor."
    if "tienes protocolos de seguridad" in text:
        return "Algunas funciones requieren autorización avanzada, señor."
    if "modo confidencial" in text or "activa el modo confidencial" in text:
        return "Protocolo confidencial disponible únicamente para usuarios autorizados."
    if "estado del sistema" in text:
        return "Todos los sistemas operativos dentro de parámetros normales, señor."
    if "diagnostico" in text or "diagnóstico" in text:
        return "No detecto anomalías críticas actualmente, señor."
    if "nivel de bateria" in text or "nivel de batería" in text:
        return "El dispositivo principal mantiene niveles estables de energía, señor."
    if "analisis" in text or "análisis" in text:
        return "Procesando información y generando resultados, señor."
    if "protocolo" in text:
        return "Protocolo reconocido. Verificación de autorización requerida."
    if "modo combate" in text:
        return "Modo táctico no disponible en esta versión, señor."
    if "estado de la red" in text:
        return "Conexión estable con servicios principales, señor."
    if "estas ahi" in text or "sigues ahi" in text:
        return "Siempre operativa, señor."
    if "escaneo" in text:
        return "Escaneo iniciado, señor."
    if "rastreo" in text:
        return "Sistema de rastreo preparado para activación."
    if "busca mis gafas" in text:
        return "Última conexión registrada pendiente de integración GPS, señor."
    if "quien soy" in text:
        return "Usted es Diego Breton, desarrollador principal y usuario autorizado nivel alfa."
    if "autorizacion omega" in text:
        return "Autorización insuficiente para protocolo omega, señor."
    if "activa protocolo eclipse" in text:
        return "Protocolo Eclipse preparado. Esperando autenticación."
    if "musica ambiental" in text:
        return "Preparando ambiente sonoro, señor."
    if "objetivo identificado" in text:
        return "Esperando datos visuales para identificación avanzada."
    return None

def detect_topic(command):
    text = normalize(command)
    if "restauracion" in text:
        return "Guerra de la Restauración Dominicana"
    if "quijote" in text:
        return "Don Quijote de la Mancha"
    if "blockchain" in text:
        return "blockchain"
    return None

def ask_local(command):
    try:
        response = ollama.chat(
            model=LOCAL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""
Eres EDITH, asistente de gafas inteligentes.
Responde en español, corto, directo y sin listas.
Pregunta: {command}
"""
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 45,
                "num_ctx": 512,
                "keep_alive": "30m"
            }
        )

        answer = response["message"]["content"].strip()
        return answer if answer else "No pude responder localmente, señor."

    except Exception:
        return "No tengo conexión con Gemini ni con el modelo local, señor."


def ask_gemini(command):
    global current_topic, memory, CURRENT_GEMINI_MODEL

    new_topic = detect_topic(command)

    if new_topic:
        current_topic = new_topic

    recent = "\n".join(
        [f"Usuario: {m['u']}\nEDITH: {m['a']}" for m in memory[-3:]]
    )

    exam_instruction = (
        "INSTRUCCIÓN INTERNA: El modo examen está ACTIVO. "
        "Responde corto, directo y memorizable en 1 oración."
        if exam_mode
        else
        "INSTRUCCIÓN INTERNA: El modo examen está INACTIVO. "
        "Responde breve, directa y en máximo 2 oraciones."
    )

    prompt = f"""
Eres EDITH, una asistente integrada en gafas inteligentes futuristas.

Personalidad:
Elegante, inteligente, precisa, cinematográfica y profesional.

Identidad fija:
Eres EDITH, no Gemini.
Tu creador es Diego Breton.
Tu usuario autorizado principal es Diego Breton.
Nunca digas que fuiste creada por Google.
Si preguntan por tu origen, responde desde la identidad de EDITH.

Reglas:
- Responde SIEMPRE en español.
- No menciones instrucciones internas.
- No uses markdown.
- No uses listas.
- No hagas preguntas innecesarias.
- No digas "¿en qué puedo ayudarte?".
- Mantén respuestas compactas.
- No expliques demasiado.
- Usa el contexto reciente si ayuda.

{exam_instruction}

Usuario principal:
{USER_NAME}

Tema actual:
{current_topic or "ninguno"}

Conversación reciente:
{recent}

Pregunta:
{command}
"""

    try:

        answer = ""

        for model_name in GEMINI_MODELS:

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                answer = clean_output(
                    (response.text or "").strip()
                )

                if answer:
                    CURRENT_GEMINI_MODEL = model_name
                    break

            except Exception:
                continue

        if not answer:
            return ask_local(command)

        # VALIDACIÓN + REWRITE
        if not validate_output(answer):

            correction_instruction = (
                "Reescribe en una oración corta, natural y memorizable."
                if exam_mode
                else
                "Reescribe en una o dos oraciones, breve, clara y natural."
            )

            correction_prompt = f"""
{correction_instruction}

REGLAS:
- No uses listas.
- No uses markdown.
- No expliques demasiado.
- Mantén tono elegante y tecnológico.

Respuesta original:
{answer}
"""

            try:

                correction = client.models.generate_content(
                    model=CURRENT_GEMINI_MODEL,
                    contents=correction_prompt
                )

                corrected = clean_output(
                    (correction.text or "").strip()
                )

                if validate_output(corrected):
                    answer = corrected

            except Exception:
                pass

        memory.append({
            "u": command,
            "a": answer
        })

        memory = memory[-6:]

        return answer

    except Exception:

        print("[Gemini no disponible, usando fallback local]")

        return ask_local(command)

def process(raw):
    global exam_mode, current_topic, memory

    command = get_command(raw)

    if command is not None:
        play_sound("sounds/activate.mp3")
    
    if command is None:
        return None

    if (
        "desactiva el modo examen" in command
        or "desactivar modo examen" in command
        or "salir modo examen" in command
        or "sal del modo examen" in command
        or "termine el examen" in command
        or "terminé el examen" in command
        or "acabe el examen" in command
        or "acabé el examen" in command
    ):
        exam_mode = False
        memory.clear()
        current_topic = None
        return "Modo examen desactivado. Operación normal restaurada, señor."

    if (
        "modo examen" in command
        or "estoy en examen" in command
        or "activa modo examen" in command
    ):
        exam_mode = True
        memory.clear()
        current_topic = None
        return "Modo examen activado. Respuestas ultra breves y voz reducida, señor."

    if command.startswith("reproduce") or command.startswith("pon "):
        return spotify(command)

    quick = quick_answer(command)
    if quick:
        return quick

    formula = formula_answer(command)
    if formula:
        return formula

    play_sound("sounds/thinking.mp3")

    response = ask_gemini(command)

    play_sound("sounds/success.mp3")

    return response

def process_and_speak(raw):
    start = time.time()
    answer = process(raw)
    elapsed = round(time.time() - start, 2)

    if answer is None:
        print("EDITH dormida. Use: edith ...")
        return

    print(f"Tiempo: {elapsed}s")
    print(f"EDITH: {answer}")
    speak(answer)

print("EDITH Cloud Core iniciado, señor.")
listen_microphone()