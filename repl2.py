import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

from synthesizer.hparams import hparams
from copy import copy

hparams.lws_mode = "speech"

hp = copy(hparams)
hp.sample_rate = 44100//2
RES = 256


def hpr(**kws):
    hp = copy(hparams)
    for k, v in kws.items():
      if not hasattr(hp, k):
        raise Exception("no such key {}".format(k))
      setattr(hp, k, v)
    return hp

def hpr_info(hp):
    print('win_size', hp.win_size)
    print('n_fft', hp.n_fft)
    print('hop_size', get_hop_size(hp))
    print('lws_mode', hp.lws_mode)
    return hp

def hpr_all(hparams):
    import types
    for k, v in [(x, getattr(hparams, x)) for x in dir(hparams) if not x.startswith('_') and type(getattr(hparams, x)) not in [types.MethodType, type(lambda x: x)]]:
        print(k, v)
    return hparams

    
num_generated = 0

def save_wav(wav, hparams):
    global num_generated
    # Save it on the disk
    fpath = "demo_output_%02d.wav" % num_generated
    while os.path.isfile(fpath):
        num_generated += 1
        fpath = "demo_output_%02d.wav" % num_generated
    print(wav.dtype)
    librosa.output.write_wav(fpath, wav.astype(np.float32), hparams.sample_rate)
    num_generated += 1
    print("\nSaved output as %s\n\n" % fpath)
    return wav

def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode=hparams.lws_mode)

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

def draw_spec(spec, title="mel spectrogram", fname="foo.png", dpi=None, cmap=None, f=None):
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
    if f:
        img = f(img)
    img.save('foo.png')
    return spec


def crop(img):
    w, h = img.size
    return img.crop([0, h//2, w, h])

# focus_measures.ContrastMeasures().fm(draw_spec(melspectrogram(twilight1, hpr(use_lws=False, lws_mode="music", num_mels=RES, sample_rate=twilight_sr, n_fft=RES*24, win_size=int(RES*0.5), hop_size=RES//32))), 'TENENGRAD1',7,0).mean()
