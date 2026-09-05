# -*- coding: utf-8 -*-
"""Separacion de voz con Mel-Band RoFormer.

QUE ES Y DE DONDE SALE
----------------------
`mel_band_roformer.py` y `mel_converter.py` NO son codigo de este proyecto. Se
copian de KimberleyJensen/Mel-Band-Roformer-Vocal-Model, que a su vez deriva del
BS-RoFormer de lucidrains, y llevan su cabecera de origen intacta. Se vendorizan
en vez de importarlos porque no existen en PyPI con estos nombres de parametro:
el checkpoint solo carga contra ESTA implementacion.

Los pesos tampoco se rehospedan. Se descargan de Kijai/MelBandRoFormer_comfy
fijando la revision, igual que el captioner descarga Qwen3-VL: rehospedar pesos
ajenos obliga a cargar con su licencia, y fijar la revision da el mismo fichero
para siempre sin copiarlo.

PARA QUE SIRVE AQUI
-------------------
Un dataset de voz sacado de material real -- una pelicula, una entrevista --
trae musica y efectos encima del dialogo. Entrenar un LoRA de timbre con eso
clona la mezcla, no la voz. Esto separa la voz antes de trocear.

Y UNA ADVERTENCIA QUE CONVIENE LEER ANTES DE USARLO
---------------------------------------------------
La separacion es DESTRUCTIVA. Lo que sale no es la voz limpia: es la voz menos
lo que el modelo creyo que era musica. Deja resonancias en las eses y en los
transitorios, y un LoRA entrenado con eso aprende tambien ese caracter.
Para material que ya viene limpio, separar EMPEORA el resultado. Tampoco quita
reverberacion: separa fuentes, no arregla la sala.

Vocal separation with Mel-Band RoFormer. The two model files are NOT this
project's code: they come from KimberleyJensen/Mel-Band-Roformer-Vocal-Model,
itself derived from lucidrains' BS-RoFormer, with their source headers intact.
They are vendored rather than imported because no PyPI package carries these
parameter names, and the checkpoint only loads against THIS implementation. The
weights are not rehosted either: they download from Kijai/MelBandRoFormer_comfy
at a pinned revision, the same way the captioner downloads Qwen3-VL.

Separation is DESTRUCTIVE: what comes out is the voice minus whatever the model
judged to be music, with artefacts on sibilants and transients that a LoRA will
learn along with the timbre. On already-clean material it makes things worse.
It does not remove reverb either -- it separates sources, not rooms.
"""
