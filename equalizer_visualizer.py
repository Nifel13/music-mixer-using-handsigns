from pyo import *
import time

# Usa la configuración que te funcionó
s = Server().boot()
s.start()

# Fuente de audio
sf = SfPlayer("audios/drugs.wav", speed=1.0, loop=True)

# Análisis FFT y cambio de tono
fft = PVAnal(sf, size=1024)
pitch = PVTranspose(fft, transpo=1)

# Síntesis con control de volumen
synth = PVSynth(pitch)
vol = Sig(0.5)  # volumen inicial
amp = synth * vol
amp.out()

# Cambios durante la ejecución
print("🎧 Reproduciendo: speed=1.0, pitch=0, vol=0.5")
time.sleep(3)

print("⚡ Subiendo velocidad a 1.5 (tempo + tono)")
sf.speed = 1.5
time.sleep(3)

print("🎵 Cambiando pitch a -4 semitonos")
pitch.transpo = 2
time.sleep(3)

print("🔉 Bajando volumen a 0.2")
vol.value = 0.2
time.sleep(3)

print("🔁 Restaurando valores originales")
sf.speed = 1.0
pitch.transpo = 1
vol.value = 1.0
time.sleep(4)

s.stop()
s.shutdown()
