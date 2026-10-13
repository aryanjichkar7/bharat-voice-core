/**
 * Bharat Online Cabs - Enterprise AudioWorklet PCM Recorder Processor
 *
 * Capabilities:
 * 1. Multi-channel to mono downmixing (handles stereo/array mics cleanly).
 * 2. Anti-aliasing box averaging downsampler from arbitrary browser sampleRate (44.1k/48k) to 16,000 Hz.
 * 3. Fractional phase continuity across 128-sample Web Audio blocks (0 dropped samples).
 * 4. Bounded 64ms chunks (1024 Int16 samples) for sub-100ms VAD response.
 * 5. Periodic numeric RMS & Peak telemetry (no raw audio saved).
 */
class PcmRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.buffer = [];
    this.chunkSize = 1024; // 1024 samples = 64 ms at 16 kHz for low-latency VAD
    this.phase = 0;
    this.isPlaybackMuted = false;

    // Listen for playback mute state from main thread
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'SET_PLAYBACK_MUTE') {
        this.isPlaybackMuted = Boolean(e.data.muted);
        if (this.isPlaybackMuted) {
          this.buffer = [];
        }
      }
    };

    // Metering telemetry counters
    this.meterSamplesCount = 0;
    this.meterSumSquares = 0;
    this.meterPeak = 0;
  }

  process(inputs, outputs, parameters) {
    if (this.isPlaybackMuted) {
      this.buffer = [];
      return true;
    }
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    const numChannels = input.length;
    const blockLength = input[0].length; // typically 128
    const sourceRate = sampleRate; // Web Audio AudioWorkletGlobalScope global

    if (sourceRate === this.targetSampleRate) {
      // 1:1 pass-through with channel mixing
      for (let i = 0; i < blockLength; i++) {
        let mono = 0;
        for (let c = 0; c < numChannels; c++) {
          mono += input[c][i];
        }
        mono /= numChannels;
        const clamped = Math.max(-1.0, Math.min(1.0, mono));
        const val = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
        this.buffer.push(val);
        this._updateMeter(mono);
      }
    } else {
      // Anti-aliasing box-filtered downsampling with phase continuity
      const ratio = sourceRate / this.targetSampleRate;
      while (this.phase < blockLength) {
        const startIdx = Math.floor(this.phase);
        const endIdx = Math.min(blockLength, Math.floor(this.phase + ratio));

        let accum = 0;
        let count = 0;
        for (let idx = startIdx; idx < endIdx; idx++) {
          let mono = 0;
          for (let c = 0; c < numChannels; c++) {
            mono += input[c][idx];
          }
          accum += mono / numChannels;
          count++;
        }

        const sample = count > 0 ? accum / count : 0;
        const clamped = Math.max(-1.0, Math.min(1.0, sample));
        const val = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
        this.buffer.push(val);
        this._updateMeter(sample);

        this.phase += ratio;
      }
      this.phase -= blockLength;
    }

    // Flush ready chunks over the port
    while (this.buffer.length >= this.chunkSize) {
      const chunk = new Int16Array(this.buffer.splice(0, this.chunkSize));
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }

    // Emit periodic telemetry meter (~every 16000 samples = ~1 sec)
    if (this.meterSamplesCount >= 16000) {
      const rms = Math.sqrt(this.meterSumSquares / this.meterSamplesCount);
      const rmsDb = rms > 0 ? 20 * Math.log10(rms) : -100;
      this.port.postMessage({
        type: 'MIC_METER',
        sourceRate: sourceRate,
        targetRate: this.targetSampleRate,
        rmsDb: Math.round(rmsDb * 10) / 10,
        peak: Math.round(this.meterPeak * 1000) / 1000
      });
      this.meterSamplesCount = 0;
      this.meterSumSquares = 0;
      this.meterPeak = 0;
    }

    return true;
  }

  _updateMeter(val) {
    this.meterSumSquares += val * val;
    this.meterSamplesCount++;
    const absVal = Math.abs(val);
    if (absVal > this.meterPeak) {
      this.meterPeak = absVal;
    }
  }
}

registerProcessor('pcm-recorder-processor', PcmRecorderProcessor);
