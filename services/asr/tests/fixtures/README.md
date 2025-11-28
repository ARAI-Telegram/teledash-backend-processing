# ASR Test Fixtures

This directory contains static test audio files used for testing the ASR service.

## Test Audio Files

### `sample_audio.wav` - Silent Audio

A 3-second silent audio file used for basic transcription tests.

- **Format**: WAV (PCM 16-bit)
- **Duration**: 3 seconds
- **Sample rate**: 16000 Hz
- **Channels**: Mono
- **Size**: ~94KB
- **Content**: Silence
- **Fixture**: `sample_audio_file`

### `sample_speech.wav` - Speech Audio

A ~4-second audio file with actual English speech for testing transcription accuracy.

- **Format**: WAV (PCM 16-bit)
- **Duration**: ~3-4 seconds
- **Sample rate**: 16000 Hz
- **Channels**: Mono
- **Size**: ~130KB
- **Content**: "Hello, this is a test for automatic speech recognition."
- **Voice**: US English (Alex or Samantha voice on macOS)
- **Fixture**: `sample_speech_file`
- **Note**: Language detection may vary; test validates transcription works, not exact language detection

Both files are automatically available via pytest fixtures in `conftest.py`. No manual setup is required.

## Regenerating Audio Files (Optional)

If you need to regenerate the test audio files:

### Silent Audio (`sample_audio.wav`)

```bash
cd services/asr/tests/fixtures
ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 3 -acodec pcm_s16le -y sample_audio.wav
```

### Speech Audio (`sample_speech.wav`)

**On macOS** (using built-in `say` command with clearer voice):

```bash
cd services/asr/tests/fixtures
# Using Samantha voice (clearer US English pronunciation)
say -v "Samantha" -o temp.aiff "Hello, this is a test for automatic speech recognition."
ffmpeg -i temp.aiff -ar 16000 -ac 1 -acodec pcm_s16le -y sample_speech.wav
rm temp.aiff
```

**On Linux** (using `espeak` or `espeak-ng`):

```bash
cd services/asr/tests/fixtures
# Install: sudo apt-get install espeak-ng
espeak-ng "Hello, this is a test for automatic speech recognition." --stdout | \
  ffmpeg -i pipe:0 -ar 16000 -ac 1 -acodec pcm_s16le -y sample_speech.wav
```

## Adding More Test Files

To add additional test audio files:

1. Place them in this directory
2. Add fixture in `conftest.py` if needed
3. Commit to repository
4. Keep files small (< 200KB recommended)
