from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
import os

## Info & args
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                    default="encoder/saved_models/pretrained.pt",
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_dir", type=Path, 
                    default="synthesizer/saved_models/logs-pretrained/",
                    help="Directory containing the synthesizer model")
parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                    default="vocoder/saved_models/pretrained/pretrained.pt",
                    help="Path to a saved vocoder")
parser.add_argument("--low_mem", action="store_true", help=\
    "If True, the memory used by the synthesizer will be freed after each use. Adds large "
    "overhead but allows to save some GPU memory for lower-end GPUs.")
parser.add_argument("--no_sound", action="store_false", default=False, help=\
    "If True, audio won't be played.")
args = parser.parse_args()

print_args(args, parser)

if not args.no_sound:
    import sounddevice as sd
    

## Print some environment information (for debugging purposes)
print("Running a test of your configuration...\n")
if not torch.cuda.is_available():
    print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
          "for deep learning, ensure that the drivers are properly installed, and that your "
          "CUDA version matches your PyTorch installation. CPU-only inference is currently "
          "not supported.", file=sys.stderr)
    #quit(-1)

#device_id = torch.cuda.current_device()
#gpu_properties = torch.cuda.get_device_properties(device_id)
#print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#      "%.1fGb total memory.\n" % 
#      (torch.cuda.device_count(),
#       device_id,
#       gpu_properties.name,
#       gpu_properties.major,
#       gpu_properties.minor,
#       gpu_properties.total_memory / 1e9))


## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
vocoder.load_model(args.voc_model_fpath)


## Run a test
print("Testing your configuration with small inputs.")
# Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
# sampling rate, which may differ.
# If you're unfamiliar with digital audio, know that it is encoded as an array of floats 
# (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
# The sampling rate is the number of values (samples) recorded per second, it is set to
# 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond 
# to an audio of 1 second.
print("\tTesting the encoder...")
encoder.embed_utterance(np.zeros(encoder.sampling_rate))

# Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
# returns, but here we're going to make one ourselves just for the sake of showing that it's
# possible.
embed = np.random.rand(speaker_embedding_size)
# Embeddings are L2-normalized (this isn't important here, but if you want to make your own 
# embeddings it will be).
embed /= np.linalg.norm(embed)
# The synthesizer can handle multiple inputs with batching. Let's create another embedding to 
# illustrate that
embeds = [embed, np.zeros(speaker_embedding_size)]
texts = ["test 1", "test 2"]
print("\tTesting the synthesizer... (loading the model will output a lot of text)")
mels = synthesizer.synthesize_spectrograms(texts, embeds)

# The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We 
# can concatenate the mel spectrograms to a single one.
mel = np.concatenate(mels, axis=1)
# The vocoder can take a callback function to display the generation. More on that later. For 
# now we'll simply hide it like this:
no_action = lambda *args: None
print("\tTesting the vocoder...")
# For the sake of making this test short, we'll pass a short target length. The target length 
# is the length of the wav segments that are processed in parallel. E.g. for audio sampled 
# at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
# 0.5 seconds which will all be generated together. The parameters here are absurdly short, and 
# that has a detrimental effect on the quality of the audio. The default parameters are 
# recommended in general.
vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

print("All test passed! You can now synthesize speech.\n\n")
#
#
### Interactive speech generation
#print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
#      "show how you can interface this project easily with your own. See the source code for "
#      "an explanation of what is happening.\n")
#
#print("Interactive generation loop")
#num_generated = 0
#prev_in_fpath = None
#while True:
#    try:
#        # Get the reference audio filepath
#        message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
#                  "wav, m4a, flac, ...):"
#        print(message)
#        fpath = input("")
#        in_fpath = prev_in_fpath if len(fpath.strip()) <= 0 else str(Path(fpath.replace("\"", "").replace("\'", "")))
#        prev_in_fpath = in_fpath
#        
#        
#        ## Computing the embedding
#        # First, we load the wav using the function that the speaker encoder provides. This is 
#        # important: there is preprocessing that must be applied.
#        
#        # The following two methods are equivalent:
#        # - Directly load from the filepath:
#        preprocessed_wav = encoder.preprocess_wav(in_fpath)
#        # - If the wav is already loaded:
#        original_wav, sampling_rate = librosa.load(in_fpath)
#        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#        print("Loaded file succesfully")
#        
#        # Then we derive the embedding. There are many functions and parameters that the 
#        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
#        # only use this function (with its default parameters):
#        embed = encoder.embed_utterance(preprocessed_wav)
#        print("Created the embedding")
#        
#        
#        ## Generating the spectrogram
#        print("Write a sentence (+-20 words) to be synthesized:")
#        text = input("")
#        
#        # The synthesizer works in batch, so you need to put your data in a list or numpy array
#        texts = [text]
#        embeds = [embed]
#        # If you know what the attention layer alignments are, you can retrieve them here by
#        # passing return_alignments=True
#        specs = synthesizer.synthesize_spectrograms(texts, embeds)
#        spec = specs[0]
#        print("Created the mel spectrogram")
#        
#        print("High quality? [y/N]: ")
#        high_quality = True if input("") == 'y' else False
#        
#        ## Generating the waveform
#        print("Synthesizing the waveform:")
#        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
#        # spectrogram, the more time-efficient the vocoder.
#        if high_quality:
#            generated_wav = vocoder.infer_waveform(spec, target=8000, overlap=1600)
#        else:
#            generated_wav = vocoder.infer_waveform(spec, target=200, overlap=20)
#        
#        
#        ## Post-generation
#        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
#        # pad it.
#        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#            
#        # Save it on the disk
#        fpath = "demo_output_%02d.wav" % num_generated
#        while os.path.isfile(fpath):
#            num_generated += 1
#            fpath = "demo_output_%02d.wav" % num_generated
#        print(generated_wav.dtype)
#        librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 
#                                 synthesizer.sample_rate)
#        num_generated += 1
#        print("\nSaved output as %s\n\n" % fpath)
#        
#        # Play the audio (non-blocking)
#        if not args.no_sound:
#            sd.stop()
#            sd.play(generated_wav, synthesizer.sample_rate)
#        
#    except Exception as e:
#        print("Caught exception: %s" % repr(e))
#        print("Restarting\n")
    
num_generated = 0

def save_wav(wav, sample_rate=synthesizer.sample_rate):
    global num_generated
    # Save it on the disk
    fpath = "demo_output_%02d.wav" % num_generated
    while os.path.isfile(fpath):
        num_generated += 1
        fpath = "demo_output_%02d.wav" % num_generated
    print(wav.dtype)
    librosa.output.write_wav(fpath, wav.astype(np.float32), sample_rate)
    num_generated += 1
    print("\nSaved output as %s\n\n" % fpath)
    return wav

import matplotlib
import matplotlib.pyplot as plt

# plt.colorbar(fig.gca().imshow(kleiner.reshape([16,-1]), cmap=cmap), ax=ax, fraction=0.046, pad=0.04).set_clim(0, 0.30); plt.savefig('foo.png')

gridspec_kw = {"width_ratios": [1]}

current_ax = None
spec_ax = None
spec_img = None
spec_fig = None
spec_dpi = 300
spec_cmap = "gray"

def orient_spec(spec):
    spec = np.array(spec)
    if len(list(spec.shape)) == 1:
        spec1 = spec.reshape([1, -1])
    #elif spec.shape[1] == 40 or spec.shape[1] == 80:
        #spec1 = spec.T
    else:
        spec1 = spec
    return spec1

def lerp(a, b, t): return (b-a)*t + a

def norm(x): return (x - x.min()) / (x.max() - x.min())

def quant(x, bot=None, top=None, lo=None, hi=None):
    bot = bot or x.min()
    top = top or x.max()
    lo = lo or x.min()
    hi = hi or x.max()
    v = 255.999
    y = (norm(x) * v).astype('int') / v
    y = lerp(bot, top, y)
    return y

def pixels2img(pixels):
    p = np.clip(pixels, 0, 1)
    h, w = p.shape
    from PIL import Image
    # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
    img = Image.new('RGB', (w, h), "black")  # create a new black image
    pixels = img.load()  # create the pixel map
    p2=lerp(0.5, 255.5, p).astype('int')
    for i in range(w):  # for every col:
        for j in range(h):  # For every row
            c = p2[j][i]
            pixels[i, j] = (c, c, c)  # set the colour accordingly
    return img

def spec2img(spec, kind='RGB'):
    spec1 = orient_spec(spec)
    #spec1 = norm(spec1)
    spec1 = (spec1 + 4.0) / 8.0
    spec1 = np.clip(spec1, 0, 1, out=spec1)
    h, w = spec1.shape
    from PIL import Image
    # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
    img = Image.new(kind, (w, h), "black")  # create a new black image
    pixels = img.load()  # create the pixel map
    spec2=lerp(0.5, 255.5, spec1).astype('int')
    for i in range(w):  # for every col:
        for j in range(h):  # For every row
            c = spec2[j][i]
            pixels[i, j] = int(c) if kind == 'L' else (c, c, c)  # set the colour accordingly
    return img

def img_aspect(img, res):
    w, h = img.size
    res = res[:]
    if res[0] == -1:
        res[0] = int(res[1] * (w / h))
    if res[1] == -1:
        res[1] = int(res[0] * (h / w))
    return res

def img_resize(img, res):
    w, h = img.size
    res = img_aspect(img, res)
    print('%dx%d -> %dx%d' % (w, h, res[0], res[1]))
    img = img.resize(res, resample=Image.LANCZOS)
    return img

def img2spec(img, lo=-4.0, hi=4.0, res=None): #res=[512,80]):
    if type(img) is str:
        from PIL import Image
        img = Image.open(img)
    if res:
        img = img_resize(img, res)
    data = list(img.getdata())
    width, height = img.size
    pixels = np.array([data[i * width:(i + 1) * width] for i in range(height)])
    if len(pixels.shape) == 3:
        pixels = pixels[:, :, 0].reshape(pixels.shape[0:2]).astype('float32')
    else:
        pixels = pixels[:, :].reshape(pixels.shape[0:2]).astype('float32')
    pixels = (pixels + 0.5) / 255.5
    pixels = lerp(lo, hi, pixels)
    return pixels

def draw_spec(spec, title="mel spectrogram", fname="foo.png", dpi=None, cmap=None):
    dpi = dpi or spec_dpi
    cmap = cmap or spec_cmap
    global current_ax
    global spec_ax
    global spec_img
    global spec_fig
    if spec_fig is not None:
        plt.close(spec_fig)
    spec1 = orient_spec(spec)
    #spec_fig, current_ax = plt.subplots(1, 1, figsize=(10, 2.25), facecolor="#F0F0F0", gridspec_kw=gridspec_kw, frameon=False)
    figsize=(10, 2.25)
    h, w = spec1.shape
    aspect = w / h
    zoom=1
    figsize=(zoom*aspect + 10/100, zoom*h/100 + 10/100)
    spec_fig, current_ax = plt.subplots(1, 1, figsize=figsize, facecolor="#F0F0F0", gridspec_kw=gridspec_kw, frameon=False)
    #spec_fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
    spec_ax = current_ax
    spec_img = spec_ax.imshow(spec1, interpolation="none", cmap=cmap)
    spec_ax.set_aspect("equal", anchor="NW")
    spec_ax.set_title(title)
    spec_ax.set_xticks([])
    spec_ax.set_yticks([])
    spec_ax.set_position([0, 0, 1, 1])
    spec_ax.figure.canvas.draw()
    #spec_ax.figure.savefig(fname, dpi=dpi)
    #spec_ax.figure.savefig(fname)
    img=spec2img(spec)
    img.save('foo.png')
    return spec


from scipy.ndimage.filters import gaussian_filter as blur

import torch
import sys
import os
waveglow_home=os.path.normpath(os.path.join(os.getcwd(), '..', 'waveglow'))
waveglow_path=os.path.join(waveglow_home, 'waveglow_256channels.pt')
sys.path += [waveglow_home]
sys.path += [os.path.join(waveglow_home, 'tacotron2')]
from denoiser import Denoiser

waveglow = torch.load(waveglow_path, map_location='cpu')['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cpu().eval()
denoiser = Denoiser(waveglow).cpu()

def denoise(wav, strength=0.5):
    return denoiser(torch.tensor(wav).unsqueeze(0), strength).cpu().numpy()[0][0]

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30


def preprocess_wav(fpath_or_wav, source_sr = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
    ## Apply the preprocessing: normalize volume and shorten long silences 
    #wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    #wav = trim_long_silences(wav)
    return wav

def load_wav(fname):
    #return encoder.preprocess_wav(*librosa.load(fname))
    return preprocess_wav(*librosa.load(fname))

def load_speaker(fname):
    return encoder.embed_utterance(load_wav(fname))

def load_voice(fname, using_partials=False):
    return embed(load_wav(fname), using_partials=using_partials)

def load_denoise(fname, strength=0.5):
    wav, sr = librosa.load(fname)
    wav2 = denoise(wav, strength=strength)
    return wav2, sr

def map1(x, f):
    return np.array([f(y) for y in x]).reshape([1,-1])[0]

speakers = [load_speaker('kleiner1.wav')]

class Namespace:
    pass

def embed(wav, using_partials=False, **kwargs):
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = encoder.compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    # Split the utterance into partials
    frames = encoder.audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = encoder.embed_frames_batch(frames_batch)
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    if not using_partials:
        frames = encoder.audio.wav_to_mel_spectrogram(wav)
        partial_embeds = encoder.embed_frames_batch(frames[None, ...])
        raw_embed = partial_embeds[0]
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    r = Namespace()
    r.wave_slices = wave_slices
    r.mel_slices = mel_slices
    r.max_wave_length = max_wave_length
    r.wav = wav
    r.frames = frames
    r.frames_batch = frames_batch
    r.partial_embeds = partial_embeds
    r.raw_embed = raw_embed
    r.embed = embed
    return r

def synth(text, voices=None):
    voices = voices or speakers
    return synthesizer.synthesize_spectrograms([text] * len(voices), voices)

def infer(mel, target=16000, overlap=800, **kws):
    return vocoder.infer_waveform(mel, target=target, overlap=overlap, **kws)

def remap(m, lo, hi, bot=None, top=None):
    bot = bot or np.min(m)
    top = top or np.max(m)
    m2 = (m - bot) / (top - bot)
    m3 = (hi - lo) * m2 + lo
    return m3

def norm_mel(m, k=1.0, bot=0.0, top=1.0):
    lo=np.min(m)
    hi=np.max(m)
    m2 = (m - lo) / (hi - lo)
    m2b = remap(m2, bot, top)
    m3 = (1-(1-np.exp(-(1-m2b)*k)))
    m3 = remap(m3, 0.0, 1.0)
    m3 = (m3 - m3.min()) / (m3.max() - m3.min())
    m4 = (hi - lo) * m3 + lo
    return m4

def say(text, target=200, overlap=50, k=1.0, voices=None, norm=True, **kws):
    r = Namespace()
    r.mels_raw = synth(text, voices=voices)
    if norm:
        r.mels = [norm_mel(m, k) for m in r.mels_raw]
    else:
        r.mels = r.mels_raw
    r.wavs = [infer(mel, target=target, overlap=overlap, **kws) for mel in r.mels]
    r.mel = np.concatenate(r.mels, axis=1)
    r.wav = np.concatenate(r.wavs)
    r.target = target
    r.overlap = overlap
    r.kws = kws
    return r

#wav2mel = encoder.audio.wav_to_mel_spectrogram

from synthesizer.audio import melspectrogram
from synthesizer.hparams import hparams


# adjusting params to melnet paper...
hp = hparams
RES = 256; hp.num_mels = hp.hop_size = RES; hp.win_size = hp.n_fft = 6 * RES

def wav2mel(wav):
    return melspectrogram(wav, hparams)

wav2spec = wav2mel

text = 'Hello, my name is Doctor Kleyener. I work at Black Mesa research facility.'

text_kleiner = 'The ball is in play, the dice have been thrown, et cetera, et cetera.'
text_alyx = 'Try grabbing those barrels from that ledge up there. You can also pull stuff over from a distance. The primary trigger emits a charge.'
#text_barney = 'Hey mister freeman. I had a bunch of messages for you but we had a system crash about an hour ago and I\'m still trying to find your files.'
text_barney = 'I had a bunch of messages for you but we had a system crash about twenty minutes ago.'
text_barney2 = 'Hey mister freeman. Just one of those days, I guess.'

import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile

def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S, hparams):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break
    
    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold
    
    return start, end

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def linearspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
    """Converts linear spectrogram to waveform using librosa"""
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram
    S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def inv_mel_spectrogram(mel_spectrogram, hparams):
    """Converts mel spectrogram to waveform using librosa"""
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    #if _mel_basis is None:
        #_mel_basis = _build_mel_basis(hparams)
    _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    #if _inv_mel_basis is None:
        #_inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)



####################

def lerp(a, b, t):
    return (b-a)*t + a

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def quant(x, bot=None, top=None, lo=None, hi=None):
    bot = bot or x.min()
    top = top or x.max()
    lo = lo or x.min()
    hi = hi or x.max()
    v = 255.999
    y = (norm(x) * v).astype('int') / v
    y = lerp(bot, top, y)
    return y

