/**
 * Bharat Online Cabs - AudioWorklet PCM Recorder Processor
 * Streams 16 kHz 16-bit linear PCM directly off the Web Audio render thread.
 */
class PcmRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.buffer = [];
    this.chunkSize = 2048; // ~128ms of audio at 16kHz for responsive VAD
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];
    const sourceRate = sampleRate; // Global AudioWorkletGlobalScope sampleRate

    if (sourceRate === this.targetSampleRate) {
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        this.buffer.push(s < 0 ? s * 0x8000 : s * 0x7FFF);
      }
    } else {
      // Linear interpolation / downsampling
      const ratio = sourceRate / this.targetSampleRate;
      const targetLength = Math.floor(channelData.length / ratio);
      for (let i = 0; i < targetLength; i++) {
        const srcIdx = Math.floor(i * ratio);
        if (srcIdx < channelData.length) {
          const s = Math.max(-1, Math.min(1, channelData[srcIdx]));
          this.buffer.push(s < 0 ? s * 0x8000 : s * 0x7FFF);
        }
      }
    }

    // Flush chunk when available
    while (this.buffer.length >= this.chunkSize) {
      const chunk = new Int16Array(this.buffer.splice(0, this.chunkSize));
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-recorder-processor', PcmRecorderProcessor);
