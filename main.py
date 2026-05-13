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
import threading
import soundfile as sf
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

USER_NAME = "Diego"
GEMINI_MODELS = [
    "gemini-2.0-flash",
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
last_topic = None
MEMORY_FILE = "memory.json" 
response_cache = {}
AUDIO_QUEUE = queue.Queue()

VOSK_MODEL = Model("vosk-model-small-es-0.42")

recognizer = KaldiRecognizer(
    VOSK_MODEL,
    16000
)

def save_memory():
    data = {
        "memory": memory,
        "current_topic": current_topic,
        "last_topic": last_topic,
        "response_cache": response_cache
    }

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

def load_memory():
    global memory, current_topic, last_topic, response_cache

    if not os.path.exists(MEMORY_FILE):
        return

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

            memory = data.get("memory", [])
            current_topic = data.get("current_topic")
            last_topic = data.get("last_topic")
            response_cache = data.get("response_cache", {})

    except Exception:
        memory = []
        current_topic = None
        last_topic = None
        response_cache = {}

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

def validate_output(text, intent_type="general", command=""):
    if not text:
        return False

    clean = clean_output(text)
    lowered = normalize(clean)
    words = len(clean.split())
    command_text = normalize(command)

    forbidden = [
        "modo examen",
        "en que puedo ayudarte",
        "markdown",
        "usuario:",
        "edith:",
        "hola",
        "dime",
        "que tipo",
        "para ayudarte",
        "piensa en un mundo",
        "mundo donde",
        "soy un modelo",
        "fui creada por google",
        "gemini"
    ]

    for item in forbidden:
        if item in lowered:
            return False

    if "?" in clean and intent_type in ["creative", "educational"]:
        return False

    if exam_mode:
        return 5 <= words <= 26

    if intent_type == "educational":
        if words < 10 or words > 45:
            return False

        if command_text.startswith("que significa ser"):
            if not lowered.startswith("ser "):
                return False
            if "significa" not in lowered:
                return False

        weak_starts = [
            "la serenidad",
            "la calma",
            "es aceptar",
            "es tolerar",
            "es algo",
            "es una cosa"
        ]

        if any(lowered.startswith(w) for w in weak_starts):
            return False

    if intent_type == "creative":
        if words < 12 or words > 45:
            return False

        if not lowered.startswith("podria") and not lowered.startswith("podría"):
            return False

        useful_words = [
            "sistema",
            "funcion",
            "función",
            "modo",
            "detectar",
            "analizar",
            "recordar",
            "automatizar",
            "integrar",
            "avisar",
            "guardar",
            "conectar",
            "leer",
            "controlar"
        ]

        if not any(w in lowered for w in useful_words):
            return False

    return words <= 55

def repair_answer_with_gemini(command, bad_answer, intent_type):
    if intent_type == "creative":
        repair_prompt = f"""
Corrige esta respuesta para que suene como EDITH, una IA integrada en gafas inteligentes.

El usuario pidió una idea futurista para EDITH.
Debes proponer UNA función concreta, útil y aplicable al proyecto real.
No saludes.
No hagas preguntas.
No uses listas.
Máximo 2 oraciones.

Respuesta mala:
{bad_answer}

Pregunta original:
{command}
"""
    elif intent_type == "educational":
        repair_prompt = f"""
Corrige esta respuesta escolar.

Debe ser una definición completa, correcta y clara.
No uses frases sueltas.
No uses listas.
No uses markdown.
Máximo 2 oraciones.

Respuesta mala:
{bad_answer}

Pregunta original:
{command}
"""
    else:
        repair_prompt = f"""
Corrige esta respuesta.

Debe ser breve, clara, natural y útil.
No saludes.
No hagas preguntas innecesarias.
No uses listas.
Máximo 2 oraciones.

Respuesta mala:
{bad_answer}

Pregunta original:
{command}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=repair_prompt
        )

        return clean_output((response.text or "").strip())

    except Exception:
        return bad_answer

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
    if "circunferencia" in text and ("que es" in text or "define" in text or "significa" in text):
        return "Una circunferencia es el conjunto de puntos de un plano que están a la misma distancia de un punto llamado centro."
    if "circulo" in text and ("que es" in text or "define" in text or "significa" in text):
        return "Un círculo es la región interior limitada por una circunferencia."
    if "radio" in text and ("que es" in text or "define" in text or "significa" in text):
        return "El radio es la distancia del centro de la circunferencia a cualquiera de sus puntos."
    if "circunferencia" in text and ("que es" in text or "define" in text or "significa" in text):
        return "Una circunferencia es el conjunto de puntos de un plano que están a la misma distancia de un punto llamado centro."
    if "circulo" in text and ("que es" in text or "define" in text or "significa" in text):
        return "Un círculo es la región interior limitada por una circunferencia."
    if "radio" in text and ("que es" in text or "define" in text or "significa" in text):
        return "El radio es la distancia del centro de la circunferencia a cualquiera de sus puntos."
    if "velocidad" in text and ("formula" in text or "ecuacion" in text):
        return "Velocidad igual distancia entre tiempo."
    if "densidad" in text and ("formula" in text or "ecuacion" in text):
        return "Densidad igual masa entre volumen."
    if "circunferencia" in text and ("formula" in text or "ecuacion" in text):
        return "Ecuación canónica: x menos h al cuadrado más y menos k al cuadrado igual radio al cuadrado."
    if "restauracion" in text:
        if exam_mode:
            return "Guerra dominicana contra España para recuperar la independencia, 1863-1865."
        return "La Guerra de la Restauración Dominicana fue un conflicto entre 1863 y 1865 para recuperar la independencia tras la anexión a España."
        
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

    if "paciencia" in text or "paciente" in text:
        return "la paciencia"
    if "estoico" in text or "estoicismo" in text:
        return "estoicismo"

    return None

def rewrite_followup(command):
    global current_topic, last_topic

    text = normalize(command)

    topic = current_topic or last_topic

    if not topic:
        return command

    followup_phrases = [
        "sigue",
        "sigue con",
        "sigue hablandome",
        "hablame mas",
        "explica mas",
        "continua",
        "dime mas",
        "y eso",
        "y que mas"
    ]

    if any(text.startswith(p) for p in followup_phrases):
        return f"Continúa explicando el tema: {topic}. Pregunta del usuario: {command}"

    return command

def make_cache_key(command_for_ai):
    mode = "exam" if exam_mode else "normal"
    topic = normalize(current_topic or "none")
    command_key = normalize(command_for_ai)
    return f"{mode}:{topic}:{command_key}"

def is_educational_question(command):
    text = normalize(command)

    starters = [
        "que es",
        "que significa",
        "quien es",
        "define",
        "explica",
        "hablame",
        "cual es",
        "como se calcula",
        "como funciona",
        "diferencia entre"
    ]

    school_words = [
        "matematica",
        "matematicas",
        "quimica",
        "fisica",
        "biologia",
        "historia",
        "geografia",
        "literatura",
        "circunferencia",
        "circulo",
        "radio",
        "diametro",
        "gas",
        "presion",
        "volumen",
        "temperatura",
        "mol",
        "restauracion",
        "blockchain",
        "virtud",
        "filosofia",
        "filosofía",
        "estoico",
        "estoicismo",
        "paciencia"
    ]

    return any(text.startswith(s) for s in starters) or any(w in text for w in school_words)


def is_creative_question(command):
    text = normalize(command)

    creative_words = [
        "idea",
        "futurista",
        "imagina",
        "crea",
        "diseña",
        "disena",
        "inventa",
        "propon",
        "que podrias hacer"
    ]

    return any(w in text for w in creative_words)

def edith_validate_contract(answer, intent_type, command):
    if not answer:
        return False

    text = clean_output(answer)
    lowered = normalize(text)
    words = len(text.split())
    command_text = normalize(command)

    forbidden = [
        "hola",
        "dime",
        "que tipo",
        "para ayudarte",
        "piensa en un mundo",
        "mundo donde",
        "gemini",
        "google",
        "modelo de lenguaje",
        "edith:",
        "usuario:",
        "markdown"
    ]

    for bad in forbidden:
        if bad in lowered:
            return False

    if exam_mode:
        return 5 <= words <= 26

    if intent_type == "educational":
        if words < 10 or words > 45:
            return False

        if command_text.startswith("que significa ser"):
            if not lowered.startswith("ser "):
                return False
            if "significa" not in lowered:
                return False

        weak_starts = [
            "la serenidad",
            "la calma",
            "la paciencia",
            "es aceptar",
            "es tolerar",
            "es algo",
            "es una cosa"
        ]

        if any(lowered.startswith(w) for w in weak_starts):
            return False

    if intent_type == "creative":
        if words < 12 or words > 45:
            return False

        if not lowered.startswith("podria") and not lowered.startswith("podría"):
            return False

        required = [
            "sistema",
            "funcion",
            "función",
            "modo",
            "detectar",
            "analizar",
            "recordar",
            "guardar",
            "automatizar",
            "integrar",
            "controlar",
            "avisar"
        ]

        if not any(w in lowered for w in required):
            return False

    return words <= 55


def edith_repair_answer(command, bad_answer, intent_type):
    command_text = normalize(command)

    if intent_type == "educational":
        if command_text.startswith("que significa ser"):
            repair_prompt = f"""
Responde SOLO la pregunta original.

Reglas obligatorias:
- Responde en español.
- Empieza exactamente con "Ser".
- Incluye la palabra "significa".
- Da una definición completa y correcta.
- No uses listas.
- No uses markdown.
- No saludes.
- Máximo 2 oraciones.

Pregunta original:
{command}
"""
        else:
            repair_prompt = f"""
Responde SOLO la pregunta original.

Reglas obligatorias:
- Responde en español.
- Da una definición completa, correcta y clara.
- No uses listas.
- No uses markdown.
- No saludes.
- Máximo 2 oraciones.

Pregunta original:
{command}
"""

    elif intent_type == "creative":
        repair_prompt = f"""
El usuario pidió una idea futurista para mejorar EDITH.

Reglas obligatorias:
- Responde en español.
- Empieza exactamente con "Podría".
- Propón UNA función concreta, útil y aplicable al proyecto EDITH.
- No saludes.
- No preguntes.
- No uses listas.
- No uses markdown.
- Máximo 2 oraciones.

Pregunta original:
{command}
"""
    else:
        repair_prompt = f"""
Responde SOLO la pregunta original en español.
Sé breve, claro y útil.
No saludes.
No uses listas.
No uses markdown.
Máximo 2 oraciones.

Pregunta original:
{command}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=repair_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=90
            )
        )

        return clean_output((response.text or "").strip())

    except Exception:
        return ""

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
No digas que eres Gemini ni Google.
Tu creador es Diego Breton.
Pregunta: {command}
"""
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 60,
                "num_ctx": 512,
                "keep_alive": "30m"
            }
        )

        answer = response["message"]["content"].strip()
        return clean_output(answer) if answer else "No pude responder localmente, señor."

    except Exception as e:
        print(f"[OLLAMA ERROR]: {e}")
        return "No tengo conexión con Gemini ni con el modelo local, señor."

def ask_local_contract(command, intent_type):
    if intent_type == "educational":
        prompt = f"""
Eres EDITH, asistente educativa precisa.

Responde SOLO la pregunta.
No saludes.
No uses listas.
No uses markdown.
Máximo 2 oraciones.

Si la pregunta es "qué significa ser X", responde con:
"Ser X significa..."

Pregunta:
{command}
"""
    elif intent_type == "creative":
        prompt = f"""
Eres EDITH, IA integrada en gafas inteligentes.

Propón UNA función concreta para mejorar EDITH.
Empieza exactamente con "Podría".
No saludes.
No preguntes.
No uses listas.
Máximo 2 oraciones.

Pregunta:
{command}
"""
    else:
        prompt = f"""
Eres EDITH, asistente de gafas inteligentes.
Responde en español, breve, directo y sin listas.
No digas que eres Gemini ni Google.
Tu creador es Diego Breton.

Pregunta:
{command}
"""

    try:
        response = ollama.chat(
            model=LOCAL_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.0,
                "num_predict": 80,
                "num_ctx": 512,
                "keep_alive": "30m"
            }
        )

        answer = response["message"]["content"].strip()
        return clean_output(answer)

    except Exception as e:
        print(f"[OLLAMA CONTRACT ERROR]: {e}")
        return ""

def ask_gemini(command):
    global current_topic
    global last_topic
    global memory
    global CURRENT_GEMINI_MODEL
    global response_cache

    new_topic = detect_topic(command)

    if new_topic and new_topic != current_topic:
        current_topic = new_topic
        memory.clear()

    if new_topic:
        last_topic = new_topic

    command_for_ai = rewrite_followup(command)
    cache_key = make_cache_key(command_for_ai)

    if cache_key in response_cache:
        return response_cache[cache_key]

    recent = "\n".join(
        [f"Usuario: {m['u']}\nEDITH: {m['a']}" for m in memory[-3:]]
    )

    intent_type = "general"

    if exam_mode:
        intent_type = "exam"

        final_prompt = f"""
Eres EDITH, asistente de estudio rápido.

Responde en español.
Da una respuesta correcta, memorizable y directa.
Máximo 18 palabras.
No uses listas.
No uses markdown.
No menciones instrucciones internas.

Si el usuario dice Restauración, asume la Guerra de la Restauración Dominicana.
Si el usuario no menciona país, asume República Dominicana cuando el contexto sea escolar.

Pregunta:
{command_for_ai}
"""

    elif is_creative_question(command):
        intent_type = "creative"

        final_prompt = f"""
El usuario pidió una idea futurista para mejorar EDITH.

Reglas obligatorias:
- Responde en español.
- Empieza exactamente con "Podría".
- Propón UNA función concreta, útil, técnica y aplicable al proyecto EDITH.
- No saludes.
- No preguntes.
- No uses listas.
- No uses markdown.
- Máximo 2 oraciones.

Contexto:
EDITH es una IA para gafas inteligentes, voz, escuela, GPS, memoria, bases de datos, NAS, casa inteligente y seguridad.

Pregunta:
{command_for_ai}
"""

    elif is_educational_question(command):
        intent_type = "educational"

        final_prompt = f"""
Eres EDITH, una IA integrada en gafas inteligentes.

Tarea:
Responder preguntas escolares con precisión de libro de texto.

Reglas obligatorias:
- Responde en español.
- Da una definición completa, correcta y clara.
- No respondas con frases sueltas.
- No uses listas.
- No uses markdown.
- Máximo 2 oraciones.
- No inventes.
- Si el usuario pregunta "qué significa ser X", responde obligatoriamente con la estructura: "Ser X significa...".
- Si es matemática, física, química, filosofía, literatura o historia, usa definiciones estándar.

Ejemplo:
Pregunta: qué significa ser estoico
Respuesta: Ser estoico significa mantener la calma, la razón y el autocontrol ante dificultades, aceptando lo que no depende de uno.

Tema actual:
{current_topic or "ninguno"}

Conversación reciente:
{recent}

Pregunta:
{command_for_ai}
"""

    else:
        intent_type = "general"

        final_prompt = f"""
Eres EDITH, una asistente integrada en gafas inteligentes futuristas.

Personalidad:
Elegante, inteligente, precisa y profesional.

Identidad fija:
Eres EDITH.
Tu creador es Diego Breton.
Tu usuario autorizado principal es Diego Breton.
Nunca digas que eres Gemini.
Nunca digas que fuiste creada por Google.

Reglas:
- Responde en español.
- Máximo 2 oraciones.
- No uses markdown.
- No uses listas.
- Mantén respuestas naturales.
- Si el usuario continúa un tema, sigue explicándolo.

Tema actual:
{current_topic or "ninguno"}

Conversación reciente:
{recent}

Pregunta:
{command_for_ai}
"""

    try:
        answer = ""

        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=final_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=90
                    )
                )

                answer = clean_output((response.text or "").strip())

                if answer:
                    CURRENT_GEMINI_MODEL = model_name
                    break

            except Exception:
                continue

        if not answer:
            local_answer = ask_local_contract(command_for_ai, intent_type)

            if local_answer and edith_validate_contract(local_answer, intent_type, command_for_ai):
                answer = local_answer
            else:
                return "No tengo conexión suficiente con mis sistemas de razonamiento, señor."

        if not edith_validate_contract(answer, intent_type, command_for_ai):
            repaired = edith_repair_answer(command_for_ai, answer, intent_type)

            if repaired and edith_validate_contract(repaired, intent_type, command_for_ai):
                answer = clean_output(repaired)
            else:
                local_repaired = ask_local_contract(command_for_ai, intent_type)

                if local_repaired and edith_validate_contract(local_repaired, intent_type, command_for_ai):
                    answer = clean_output(local_repaired)
                else:
                    if intent_type == "educational":
                        return "No tengo suficiente precisión para responder eso ahora mismo, señor. Recomiendo activar modo preciso."
                    elif intent_type == "creative":
                        return "No pude generar una propuesta útil con mis sistemas actuales, señor."
                    else:
                        return "No pude generar una respuesta confiable, señor."

        memory.append({
            "u": command,
            "a": answer
        })

        memory = memory[-6:]

        response_cache[cache_key] = answer

        if len(response_cache) > 100:
            first_key = next(iter(response_cache))
            del response_cache[first_key]

        save_memory()

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

def text_input_loop():

    while True:

        try:

            raw = input("\n> ").strip()

            if not raw:
                continue

            if normalize(raw) in ["salir", "exit", "apagar"]:

                print("EDITH: Apagando sistemas.")
                os._exit(0)

            process_and_speak(raw)

        except Exception as e:

            print(f"[TEXT ERROR]: {e}")

load_memory()
print("EDITH Cloud Core iniciado, señor.")
threading.Thread(
    target=text_input_loop,
    daemon=True
).start()

listen_microphone()