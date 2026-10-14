/**
 * Bharat Online Cabs - Voice Assistant Client
 * Full-duplex Web Audio capture via AudioWorklet (16kHz PCM16) & streaming playback (22.05kHz PCM16).
 * OpenStreetMap (Leaflet) cartography with responsive polyline, POI discovery, and honest dispatch telemetry.
 */

// Configuration & Default Coordinates (Nagpur, MH)
const DEFAULT_LAT = 21.1458;
const DEFAULT_LON = 79.0882;
const IN_SAMPLE_RATE = 16000;
const OUT_SAMPLE_RATE = 22050;

// Session State Machine Constants
const SESSION_STATES = {
  READY: 'READY',
  REQUESTING_MIC: 'REQUESTING_MIC',
  LISTENING: 'LISTENING',
  PROCESSING: 'PROCESSING',
  SPEAKING: 'SPEAKING',
  PAUSED: 'PAUSED',
  ERROR: 'ERROR',
  DISCONNECTED: 'DISCONNECTED',
};

const SESSION_IDLE_TIMEOUT_MS = 5000;
const SPEECH_START_TIMEOUT_MS = 8000;

// Application State
let ws = null;
let audioCaptureCtx = null;
let audioPlaybackCtx = null;
let micStream = null;
let workletNode = null;
let micProcessor = null; // fallback
let isRecording = false;
let currentSessionState = SESSION_STATES.READY;
let orbState = SESSION_STATES.READY;
let speechStartTimeout = null;
let idleListeningTimeout = null;
let processingWatchdog = null;
let listeningCycleToken = 0;
let lastCommittedTurnSeq = 0;
let interimBubble = null;
let nextPlayTime = 0;
let activeAudioSources = [];
let userLocation = { lat: DEFAULT_LAT, lon: DEFAULT_LON };
let currentLanguage = 'en'; // 'en' | 'hi' | 'mr'

function clearSpeechStartTimeout() {
  if (speechStartTimeout) {
    clearTimeout(speechStartTimeout);
    speechStartTimeout = null;
  }
}

function clearIdleListeningTimeout() {
  if (idleListeningTimeout) {
    clearTimeout(idleListeningTimeout);
    idleListeningTimeout = null;
  }
}

function clearProcessingWatchdog() {
  if (processingWatchdog) {
    clearTimeout(processingWatchdog);
    processingWatchdog = null;
  }
}

function clearAllTimeouts() {
  clearSpeechStartTimeout();
  clearIdleListeningTimeout();
  clearProcessingWatchdog();
}

function scheduleSpeechStartTimeout(expectedCycle) {
  clearSpeechStartTimeout();
  speechStartTimeout = setTimeout(() => {
    if (expectedCycle !== undefined && expectedCycle !== listeningCycleToken) {
      console.debug(`[LIFECYCLE] Stale speechStartTimeout fired for cycle ${expectedCycle} (current: ${listeningCycleToken}), ignoring`);
      return;
    }
    if (isRecording && (currentSessionState === SESSION_STATES.LISTENING || orbState === SESSION_STATES.LISTENING)) {
      console.info(`[LIFECYCLE] ${SPEECH_START_TIMEOUT_MS}ms speech-start timeout expired: stopping capture`);
      stopAudioCapture('SPEECH_START_TIMEOUT');
      setOrbState(SESSION_STATES.PAUSED, 'Timed out waiting for speech');
    }
  }, SPEECH_START_TIMEOUT_MS);
}

function scheduleIdleListeningTimeout(expectedCycle) {
  clearIdleListeningTimeout();
  idleListeningTimeout = setTimeout(() => {
    if (expectedCycle !== undefined && expectedCycle !== listeningCycleToken) {
      console.debug(`[LIFECYCLE] Stale idleListeningTimeout fired for cycle ${expectedCycle} (current: ${listeningCycleToken}), ignoring`);
      return;
    }
    if (isRecording && (currentSessionState === SESSION_STATES.LISTENING || orbState === SESSION_STATES.LISTENING)) {
      console.info(`[LIFECYCLE] ${SESSION_IDLE_TIMEOUT_MS}ms idle listening timeout expired: stopping capture`);
      stopAudioCapture('IDLE_TIMEOUT');
      setOrbState(SESSION_STATES.PAUSED, 'Paused due to inactivity');
    }
  }, SESSION_IDLE_TIMEOUT_MS);
}

function scheduleProcessingWatchdog() {
  clearProcessingWatchdog();
  processingWatchdog = setTimeout(() => {
    if (isRecording && (currentSessionState === SESSION_STATES.PROCESSING || orbState === SESSION_STATES.PROCESSING)) {
      console.warn('[WATCHDOG] Operation timed out after 15s without server response');
      stopAudioCapture('OPERATION_TIMEOUT');
      clearInterimTranscript();
      setOrbState(SESSION_STATES.ERROR, 'Search request timed out. Tap orb to retry.');
      setTripPanelError('Search request timed out. Tap the orb to retry or specify a nearby landmark.');
      appendTranscript('bot', "The search request timed out. Please tap the orb to try again or specify a nearby landmark.");
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'INTERRUPT_PLAYBACK' }));
      }
    }
  }, 15000);
}

function enterListeningState(reason) {
  if (!isRecording || currentRideState === 'RIDE_ACTIVE' || isBookingConfirmedPendingSpeech) {
    console.info(`[LIFECYCLE] enterListeningState suppressed (isRecording=${isRecording}, rideState=${currentRideState}, pendingBookingSpeech=${isBookingConfirmedPendingSpeech})`);
    return;
  }
  listeningCycleToken++;
  const cycle = listeningCycleToken;
  console.info(`[LIFECYCLE] enterListeningState (cycle=${cycle}, reason=${reason})`);
  clearProcessingWatchdog();
  clearSpeechStartTimeout();
  clearIdleListeningTimeout();
  setOrbState(SESSION_STATES.LISTENING);
  scheduleIdleListeningTimeout(cycle);
}

let clientTurnSeq = 0;
const renderedMsgIds = new Set();
let lastRenderedRole = null;
let lastRenderedText = null;
let lastRenderedTime = 0;
let isBookingConfirmedPendingSpeech = false;
let isSendingText = false;
let lastUserActivityTime = Date.now();
let activeRideData = null;
let currentRideState = 'IDLE'; // 'IDLE' | 'PROPOSAL' | 'RIDE_ACTIVE' | 'CANCELLED'

// Leaflet OpenStreetMap State
let leafletMap = null;
let leafletPickupMarker = null;
let leafletDestMarker = null;
let leafletRoutePolyline = null;
let leafletPoiMarkers = [];
let mapTileUrl = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

// DOM Elements
const voiceOrb = document.getElementById('voiceOrb');
const orbGlow = document.getElementById('orbGlow');
const orbStateTitle = document.getElementById('orbStateTitle');
const orbStateSubtitle = document.getElementById('orbStateSubtitle');
const orbTitle = orbStateTitle;
const orbSubtitle = orbStateSubtitle;

const liveStatusBadge = document.getElementById('liveStatusBadge');
const liveStatusLabel = document.getElementById('liveStatusLabel');
const gpsCoordsLabel = document.getElementById('gpsCoordsLabel');

const transcriptBox = document.getElementById('transcriptBox');
const textInput = document.getElementById('textInput');
const sendBtn = document.getElementById('sendBtn');
const clearTranscriptBtn = document.getElementById('clearTranscriptBtn');

const poiPillsSection = document.getElementById('poiPillsSection');
const poiPillsList = document.getElementById('poiPillsList');

const cardIdleView = document.getElementById('cardIdleView');
const cardProposalView = document.getElementById('cardProposalView');
const cardActiveView = document.getElementById('cardActiveView');
const cardCancelledView = document.getElementById('cardCancelledView');
const btnChangeDestination = document.getElementById('btnChangeDestination');
const btnPlanNewTrip = document.getElementById('btnPlanNewTrip');

const cardOriginName = document.getElementById('cardOriginName');
const cardDestName = document.getElementById('cardDestName');
const cardFareAmount = document.getElementById('cardFareAmount');
const cardDistance = document.getElementById('cardDistance');
const cardDuration = document.getElementById('cardDuration');
const btnConfirmRide = document.getElementById('btnConfirmRide');

const confirmedBookingId = document.getElementById('confirmedBookingId');
const confirmedDistance = document.getElementById('confirmedDistance');
const confirmedDuration = document.getElementById('confirmedDuration');
const confirmedFare = document.getElementById('confirmedFare');
const btnCancelRide = document.getElementById('btnCancelRide');

function setTripPanelError(errorMessage) {
  if (!cardIdleView) return;
  const statusBadge = cardIdleView.querySelector('.trip-status-badge');
  if (statusBadge) {
    statusBadge.className = 'trip-status-badge badge-error';
    statusBadge.textContent = 'Search Error';
  }
  const guidanceTitle = cardIdleView.querySelector('.guidance-text strong');
  if (guidanceTitle) {
    guidanceTitle.textContent = 'Search Request Timed Out';
  }
  const guidanceDesc = cardIdleView.querySelector('.guidance-text p');
  if (guidanceDesc) {
    guidanceDesc.textContent = errorMessage || 'The search service is taking longer than expected. Tap the orb to retry or specify a nearby landmark.';
  }
}

function clearTripPanelError() {
  if (!cardIdleView) return;
  const statusBadge = cardIdleView.querySelector('.trip-status-badge');
  if (statusBadge) {
    statusBadge.className = 'trip-status-badge badge-idle';
    statusBadge.textContent = 'Awaiting Destination';
  }
  const guidanceTitle = cardIdleView.querySelector('.guidance-text strong');
  if (guidanceTitle) {
    guidanceTitle.textContent = 'Start a Trip in Nagpur';
  }
  const guidanceDesc = cardIdleView.querySelector('.guidance-text p');
  if (guidanceDesc) {
    guidanceDesc.textContent = 'Speak your destination aloud or type below. The system will compute the road route, driving time, and fare quotation.';
  }
}

// ----------------------------------------------------------------------------- 1. OpenStreetMap (Leaflet) Initialization
async function loadMapEngine() {
  try {
    const res = await fetch('/api/config');
    const config = await res.json();
    mapTileUrl = config.map_tile_url || 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

    if (config.default_lat && config.default_lon) {
      userLocation.lat = config.default_lat;
      userLocation.lon = config.default_lon;
    }
  } catch (err) {
    console.error('Error fetching /api/config:', err);
  }

  initLeafletMap();

  // Attempt browser GPS acquisition with location honesty
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        userLocation.lat = pos.coords.latitude;
        userLocation.lon = pos.coords.longitude;
        userLocation.is_exact_gps = true;
        if (gpsCoordsLabel) {
          gpsCoordsLabel.textContent = `GPS (${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E)`;
        }
        if (leafletMap && leafletPickupMarker) {
          leafletPickupMarker.setLatLng([userLocation.lat, userLocation.lon]);
          leafletMap.setView([userLocation.lat, userLocation.lon]);
        }
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'CLIENT_INIT',
            lat: userLocation.lat,
            lon: userLocation.lon,
            lang: currentLanguage,
            gps_source: 'client_exact'
          }));
        }
      },
      (err) => {
        console.info('Using default dispatch coordinates (GPS permission not granted):', err.message);
        userLocation.is_exact_gps = false;
        if (gpsCoordsLabel) {
          gpsCoordsLabel.textContent = `Dispatch Core (Approximate: ${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E)`;
        }
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'CLIENT_INIT',
            lat: userLocation.lat,
            lon: userLocation.lon,
            lang: currentLanguage,
            gps_source: 'fallback_approximate'
          }));
        }
      },
      { timeout: 7000, enableHighAccuracy: true }
    );
  }
}

function initLeafletMap() {
  const mapElement = document.getElementById('map');
  if (!mapElement || typeof L === 'undefined') return;

  if (leafletMap) {
    leafletMap.remove();
  }

  leafletMap = L.map('map', {
    zoomControl: true,
    attributionControl: true
  }).setView([userLocation.lat, userLocation.lon], 14);

  // Standard OpenStreetMap Tile Layer with visible attribution
  L.tileLayer(mapTileUrl, {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors'
  }).addTo(leafletMap);

  // Pickup Marker (Green Beacon)
  const pickupIcon = L.divIcon({
    className: 'custom-leaflet-pickup',
    html: `
      <div style="position: relative; width: 28px; height: 28px;">
        <div style="position: absolute; width: 28px; height: 28px; border-radius: 50%; background: rgba(16, 185, 129, 0.35);"></div>
        <div style="position: absolute; top: 5px; left: 5px; width: 18px; height: 18px; border-radius: 50%; background: #10b981; border: 2.5px solid #ffffff; box-shadow: 0 2px 6px rgba(0,0,0,0.3);"></div>
      </div>
    `,
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });

  leafletPickupMarker = L.marker([userLocation.lat, userLocation.lon], {
    icon: pickupIcon,
    title: 'Pickup Location'
  }).addTo(leafletMap);

  setTimeout(() => {
    leafletMap.invalidateSize();
  }, 250);
}

// ----------------------------------------------------------------------------- 2. Status Handshake
function setConnectionStatus(connected) {
  if (connected) {
    if (liveStatusBadge) {
      liveStatusBadge.className = 'live-status-pill connected';
    }
    if (liveStatusLabel && liveStatusLabel.textContent !== 'Active Ride Monitoring') {
      liveStatusLabel.textContent = 'Voice Core Active';
    }
  } else {
    if (liveStatusBadge) {
      liveStatusBadge.className = 'live-status-pill';
    }
    if (liveStatusLabel) {
      liveStatusLabel.textContent = 'Offline';
    }
  }
}

// ----------------------------------------------------------------------------- 3. WebSocket Transport
function getWsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/voice?lat=${userLocation.lat}&lon=${userLocation.lon}&lang=${currentLanguage}`;
}

function connectWebSocket() {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      resolve(ws);
      return;
    }

    setOrbState(SESSION_STATES.REQUESTING_MIC, 'Initializing secure voice session');

    ws = new WebSocket(getWsUrl());
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus(true);
      ws.send(JSON.stringify({
        type: 'CLIENT_INIT',
        lat: userLocation.lat,
        lon: userLocation.lon,
        lang: currentLanguage
      }));
      resolve(ws);
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        handleServerTextMessage(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        handleServerAudioBuffer(event.data);
      } else if (event.data instanceof Blob) {
        event.data.arrayBuffer().then(buf => handleServerAudioBuffer(buf)).catch(err => console.error('[AUDIO] Blob decoding error:', err));
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      setConnectionStatus(false);
      setOrbState(SESSION_STATES.ERROR, 'Could not reach voice dispatch');
      reject(err);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnectionStatus(false);
      stopAudioCapture();
      setOrbState(SESSION_STATES.DISCONNECTED, 'Voice session disconnected');
    };
  });
}

// ----------------------------------------------------------------------------- 4. Audio Ingestion via AudioWorklet
async function startAudioCapture() {
  try {
    setOrbState(SESSION_STATES.REQUESTING_MIC);
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    audioCaptureCtx = new AudioCtxClass();
    await audioCaptureCtx.resume();

    if (!audioPlaybackCtx) {
      audioPlaybackCtx = new AudioCtxClass({ sampleRate: OUT_SAMPLE_RATE });
      await audioPlaybackCtx.resume();
    }

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    });

    const source = audioCaptureCtx.createMediaStreamSource(micStream);

    // AudioWorklet Pipeline
    if (audioCaptureCtx.audioWorklet) {
      try {
        await audioCaptureCtx.audioWorklet.addModule('/static/pcm-recorder-worklet.js');
        workletNode = new AudioWorkletNode(audioCaptureCtx, 'pcm-recorder-processor');
        workletNode.port.onmessage = (e) => {
          if (e.data && e.data.type === 'MIC_METER') {
            if (e.data.rmsDb < -55) {
              console.warn(`[MIC WARNING] Low input level (${e.data.rmsDb} dB). Please speak clearly or check mic sensitivity.`);
            } else {
              console.log(`[MIC METER] ${e.data.sourceRate}Hz -> 16kHz | RMS: ${e.data.rmsDb} dB | Peak: ${e.data.peak}`);
            }
            return;
          }

          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };

        source.connect(workletNode);
        // Connect to a muted GainNode to guarantee continuous Web Audio graph rendering
        const muteGain = audioCaptureCtx.createGain();
        muteGain.gain.value = 0;
        workletNode.connect(muteGain);
        muteGain.connect(audioCaptureCtx.destination);

        isRecording = true;
        enterListeningState('AUDIO_WORKLET_READY');
        console.info(`[AUDIO PIPELINE] AudioWorklet active at native ${audioCaptureCtx.sampleRate} Hz -> 16000 Hz`);
        return;
      } catch (workletErr) {
        console.warn('AudioWorklet module failed, falling back to ScriptProcessor:', workletErr);
      }
    }

    // Fallback ScriptProcessor
    const bufferSize = 2048;
    micProcessor = audioCaptureCtx.createScriptProcessor(bufferSize, 1, 1);
    micProcessor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const inputFloat32 = e.inputBuffer.getChannelData(0);
      const resampled16k = downsampleBuffer(inputFloat32, audioCaptureCtx.sampleRate, IN_SAMPLE_RATE);
      const pcm16 = float32ToInt16(resampled16k);
      ws.send(pcm16.buffer);
    };

    source.connect(micProcessor);
    micProcessor.connect(audioCaptureCtx.destination);
    isRecording = true;
    enterListeningState('SCRIPT_PROCESSOR_READY');
  } catch (err) {
    console.error('Microphone access error:', err);
    setOrbState(SESSION_STATES.ERROR, 'Microphone permission denied or device unavailable');
  }
}

function stopAudioCapture(reason) {
  console.info(`[LIFECYCLE] stopAudioCapture called (reason: ${reason || 'unspecified'})`);
  clearAllTimeouts();
  if (workletNode) {
    try { workletNode.disconnect(); } catch (e) {}
    workletNode = null;
  }
  if (micProcessor) {
    try { micProcessor.disconnect(); } catch (e) {}
    micProcessor = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(track => {
      try {
        console.info(`[LIFECYCLE] Stopping track: ${track.kind}, readyState before: ${track.readyState}`);
        track.stop();
        console.info(`[LIFECYCLE] readyState after: ${track.readyState}`);
      } catch (e) {}
    });
    micStream = null;
  }
  if (audioCaptureCtx && audioCaptureCtx.state !== 'closed') {
    try { audioCaptureCtx.close(); } catch (e) {}
    audioCaptureCtx = null;
  }
  isRecording = false;
}

function downsampleBuffer(buffer, sourceRate, targetRate) {
  if (sourceRate === targetRate) return buffer;
  const sampleRateRatio = sourceRate / targetRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function float32ToInt16(buffer) {
  const int16 = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    const s = Math.max(-1, Math.min(1, buffer[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}

// ----------------------------------------------------------------------------- 5. Audio Playback
function handleServerAudioBuffer(arrayBuffer) {
  clearAllTimeouts();
  if (!audioPlaybackCtx) {
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    audioPlaybackCtx = new AudioCtxClass({ sampleRate: OUT_SAMPLE_RATE });
  }
  if (audioPlaybackCtx.state === 'suspended') {
    audioPlaybackCtx.resume();
  }

  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / (int16[i] < 0 ? 32768 : 32767);
  }

  const audioBuffer = audioPlaybackCtx.createBuffer(1, float32.length, OUT_SAMPLE_RATE);
  audioBuffer.copyToChannel(float32, 0);

  const sourceNode = audioPlaybackCtx.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.connect(audioPlaybackCtx.destination);

  const now = audioPlaybackCtx.currentTime;
  const startTime = Math.max(now, nextPlayTime);
  sourceNode.start(startTime);
  nextPlayTime = startTime + audioBuffer.duration;

  activeAudioSources.push(sourceNode);

  // Mute microphone in worklet while assistant audio is outputting to prevent acoustic feedback
  if (workletNode) {
    workletNode.port.postMessage({ type: 'SET_PLAYBACK_MUTE', muted: true });
  }

  sourceNode.onended = () => {
    const idx = activeAudioSources.indexOf(sourceNode);
    if (idx > -1) activeAudioSources.splice(idx, 1);
    if (activeAudioSources.length === 0) {
      setTimeout(() => {
        if (activeAudioSources.length === 0) {
          if (workletNode) {
            workletNode.port.postMessage({ type: 'SET_PLAYBACK_MUTE', muted: false });
          }
          if (isBookingConfirmedPendingSpeech || currentRideState === 'RIDE_ACTIVE') {
            isBookingConfirmedPendingSpeech = false;
            console.info('[LIFECYCLE] Speech finished after booking confirmed: terminating mic capture');
            stopAudioCapture('BOOKING_COMPLETED_AFTER_SPEECH');
            setOrbState(SESSION_STATES.READY, 'Booking confirmed');
            return;
          }
          if (isRecording) {
            enterListeningState('PLAYBACK_COMPLETE');
          } else {
            setOrbState(SESSION_STATES.READY);
          }
        }
      }, 150);
    }
  };

  setOrbState(SESSION_STATES.SPEAKING);
}

function interruptPlayback() {
  activeAudioSources.forEach(src => {
    try { src.stop(); } catch (e) {}
  });
  activeAudioSources = [];
  if (audioPlaybackCtx) {
    nextPlayTime = audioPlaybackCtx.currentTime;
  }
  if (workletNode) {
    workletNode.port.postMessage({ type: 'SET_PLAYBACK_MUTE', muted: false });
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'INTERRUPT_PLAYBACK' }));
  }
  if (isRecording) {
    if (currentSessionState !== SESSION_STATES.PROCESSING && orbState !== SESSION_STATES.PROCESSING) {
      enterListeningState('PLAYBACK_INTERRUPTED');
    } else {
      console.info('[LIFECYCLE] interruptPlayback: preserving active PROCESSING state');
    }
  }
}

// ----------------------------------------------------------------------------- 6. Telemetry & Frame Handlers
function handleServerTextMessage(text) {
  try {
    const data = JSON.parse(text);
    console.log('Telemetry Frame:', data);

    // Keep turn counter synchronized
    if (data.turn_seq !== undefined) {
      clientTurnSeq = Math.max(clientTurnSeq, data.turn_seq);
    }

    // Drop stale frames for superseded turns unless it's a lifecycle confirmation
    if (data.turn_seq !== undefined && data.turn_seq < lastCommittedTurnSeq && data.type !== 'CONVERSATION_CLEARED' && data.type !== 'BOOKING_CONFIRMED') {
      console.info(`[DROPPED] Discarding obsolete server frame (${data.type}) for turn_seq ${data.turn_seq} (last committed: ${lastCommittedTurnSeq})`);
      return;
    }

    switch (data.type) {
      case 'CONNECTION_STATUS':
        if (data.status === 'CONNECTED') {
          setConnectionStatus(true);
        }
        break;

      case 'SESSION_READY':
        setConnectionStatus(true);
        if (isRecording) {
          enterListeningState('SESSION_READY');
        } else {
          setOrbState(SESSION_STATES.READY);
        }
        break;

      case 'LANGUAGE_CHANGED':
        if (data.language) {
          currentLanguage = data.language;
          updateLanguagePillsUI(data.language);
        }
        break;

      case 'INTERIM_TRANSCRIPT':
        // Live provisional transcript rendering - updates in-place without triggering search
        if (data.turn_seq !== undefined && data.turn_seq <= lastCommittedTurnSeq) {
          console.debug(`[DROPPED] Discarding stale interim transcript for turn ${data.turn_seq} (last committed: ${lastCommittedTurnSeq})`);
          return;
        }
        renderInterimTranscript(data.text, data.turn_seq);
        break;

      case 'USER_TRANSCRIPT':
        // Final transcript replaces provisional bubble cleanly
        clearProcessingWatchdog();
        scheduleProcessingWatchdog();
        if (data.turn_seq) {
          lastCommittedTurnSeq = Math.max(lastCommittedTurnSeq, data.turn_seq);
          clientTurnSeq = Math.max(clientTurnSeq, data.turn_seq);
        }
        clearInterimTranscript();
        appendTranscript('user', data.text, data.msg_id, data.turn_seq);
        recordActivity();
        setOrbState(SESSION_STATES.PROCESSING, 'Routing to destination & calculating fare...');
        break;

      case 'STT_EMPTY':
        clearProcessingWatchdog();
        clearInterimTranscript();
        if (isRecording && activeAudioSources.length === 0) {
          enterListeningState('STT_EMPTY_REARM');
        }
        break;

      case 'BOT_RESPONSE':
        clearProcessingWatchdog();
        if (data.turn_seq) {
          clientTurnSeq = Math.max(clientTurnSeq, data.turn_seq);
        }
        appendTranscript('bot', data.text, data.msg_id, data.turn_seq);
        break;

      case 'CONVERSATION_CLEARED':
        clearProcessingWatchdog();
        clearInterimTranscript();
        console.info('[TELEMETRY] Conversation cleared on server:', data);
        break;

      case 'DESTINATION_CLEARED':
        clearInterimTranscript();
        setTripCardState('idle');
        break;

      case 'PERF_DIAGNOSTICS':
        console.group(`⏱️ Turn Latency Diagnostics [${data.turn_id}]`);
        console.table(data.timings);
        console.groupEnd();
        break;

      case 'VAD_SPEECH_START':
        clearAllTimeouts();
        recordActivity();
        setOrbState(SESSION_STATES.PROCESSING, 'Listening to speech...');
        break;

      case 'VAD_SPEECH_STOP':
        setOrbState(SESSION_STATES.PROCESSING, 'Finalizing response...');
        scheduleProcessingWatchdog();
        break;

      case 'POI_CANDIDATES':
        clearTripPanelError();
        renderPoiCandidates(data.candidates || []);
        break;

      case 'MAP_ROUTE_PREVIEW':
        clearTripPanelError();
        renderRoutePreview(data);
        break;

      case 'BOOKING_CONFIRMED':
        renderBookingConfirmed(data);
        break;

      case 'RIDE_CANCELLED':
        handleRideCancelled();
        break;

      case 'INTERRUPT':
        interruptPlayback();
        break;

      case 'ERROR':
        clearProcessingWatchdog();
        console.error('Server error:', data.error);
        setTripPanelError(data.error);
        appendTranscript('bot', `Error: ${data.error}`, data.msg_id, data.turn_seq);
        setOrbState(SESSION_STATES.ERROR, data.error);
        break;

      default:
        break;
    }
  } catch (err) {
    console.error('Non-JSON text frame:', text);
  }
}

// ----------------------------------------------------------------------------- 7. UI Rendering & Map Cartography
function getLangName(code) {
  if (code === 'hi') return 'Hindi';
  if (code === 'mr') return 'Marathi';
  return 'English';
}

function updateLanguagePillsUI(lang) {
  document.querySelectorAll('.lang-pill').forEach(pill => {
    if (pill.dataset.lang === lang) {
      pill.classList.add('active');
    } else {
      pill.classList.remove('active');
    }
  });
}

function setLanguage(lang) {
  if (lang === 'gu') {
    appendTranscript('bot', 'Gujarati neural voice model is currently in development. Please select English, Hindi, or Marathi.');
    return;
  }
  currentLanguage = lang;
  updateLanguagePillsUI(lang);
  const initialBubble = document.getElementById('initialGreetingText');
  if (initialBubble) {
    const greetings = {
      'en': 'Hello! Welcome to Bharat Online Cabs. Where in Nagpur would you like to go?',
      'hi': 'नमस्ते! भारत ऑनलाइन कैब्स में आपका स्वागत है। आप नागपुर में कहाँ जाना चाहेंगे?',
      'mr': 'नमस्कार! भारत ऑनलाईन कॅब्समध्ये आपले स्वागत आहे. आपण नागपूरमध्ये कुठे जाऊ इच्छिता?'
    };
    initialBubble.textContent = greetings[lang] || greetings['en'];
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'SET_LANGUAGE',
      lang: lang
    }));
  }
}

function renderInterimTranscript(text, turnSeq) {
  if (!text || !text.trim()) return;
  if (turnSeq !== undefined && turnSeq <= lastCommittedTurnSeq) return;
  const clean = text.trim();

  if (!interimBubble || !interimBubble.parentNode) {
    const oldBubble = document.getElementById('interimTranscriptBubble');
    if (oldBubble) {
      oldBubble.remove();
    }
    interimBubble = document.createElement('div');
    interimBubble.className = 'message-bubble user interim';
    interimBubble.id = 'interimTranscriptBubble';
    interimBubble.dataset.turnSeq = (turnSeq !== undefined) ? turnSeq : '';

    const speakerTag = document.createElement('span');
    speakerTag.className = 'speaker-tag';
    speakerTag.textContent = 'You (Speaking...)';

    const content = document.createElement('p');
    content.className = 'interim-text';
    content.textContent = clean;

    interimBubble.appendChild(speakerTag);
    interimBubble.appendChild(content);
    transcriptBox.appendChild(interimBubble);
  } else {
    interimBubble.dataset.turnSeq = (turnSeq !== undefined) ? turnSeq : '';
    const content = interimBubble.querySelector('.interim-text') || interimBubble.querySelector('p');
    if (content) {
      content.textContent = clean;
    }
  }
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function clearInterimTranscript() {
  if (interimBubble && interimBubble.parentNode) {
    interimBubble.remove();
  }
  interimBubble = null;
  const oldBubble = document.getElementById('interimTranscriptBubble');
  if (oldBubble) {
    oldBubble.remove();
  }
}

function setOrbState(state, detail) {
  let s = (state || SESSION_STATES.READY).toString().toUpperCase();
  if (s === 'IDLE') s = SESSION_STATES.READY;
  if (s === 'PROCESSING_SPEECH' || s === 'PREPARING_RESPONSE') s = SESSION_STATES.PROCESSING;
  if (s === 'PAUSED_INACTIVITY') s = SESSION_STATES.PAUSED;

  currentSessionState = s;
  orbState = s;

  let cssClass = 'idle';
  switch (s) {
    case SESSION_STATES.READY:
      cssClass = 'idle';
      orbTitle.textContent = 'Tap to Speak';
      orbSubtitle.textContent = 'Voice Assistant';
      break;
    case SESSION_STATES.REQUESTING_MIC:
      cssClass = 'requesting_mic';
      orbTitle.textContent = 'Requesting Microphone...';
      orbSubtitle.textContent = 'Please allow microphone access in your browser';
      break;
    case SESSION_STATES.LISTENING:
      cssClass = 'listening';
      orbTitle.textContent = 'Listening...';
      orbSubtitle.textContent = `Speak in ${getLangName(currentLanguage)}`;
      break;
    case SESSION_STATES.PROCESSING:
      cssClass = 'processing';
      orbTitle.textContent = 'Processing...';
      orbSubtitle.textContent = detail || 'Transcribing utterance & routing';
      break;
    case SESSION_STATES.SPEAKING:
      cssClass = 'speaking';
      orbTitle.textContent = 'Disha Speaking...';
      orbSubtitle.textContent = 'Voice Concierge Response';
      break;
    case SESSION_STATES.PAUSED:
      cssClass = 'paused';
      orbTitle.textContent = 'Paused due to inactivity';
      orbSubtitle.textContent = 'Tap orb to resume hands-free voice dispatch';
      break;
    case SESSION_STATES.ERROR:
      cssClass = 'error';
      const detailLower = (detail || '').toLowerCase();
      if (detailLower.includes('search') || detailLower.includes('destination') || detailLower.includes('route') || detailLower.includes('map') || detailLower.includes('timed out')) {
        orbTitle.textContent = 'Destination Search Error';
      } else if (detailLower.includes('microphone') || detailLower.includes('audio') || detailLower.includes('device') || detailLower.includes('permission')) {
        orbTitle.textContent = 'Microphone / Audio Error';
      } else if (detailLower.includes('connection') || detailLower.includes('disconnect') || detailLower.includes('server')) {
        orbTitle.textContent = 'Connection Error';
      } else {
        orbTitle.textContent = 'System Error';
      }
      orbSubtitle.textContent = detail || 'Search or connection issue';
      break;
    case SESSION_STATES.DISCONNECTED:
      cssClass = 'disconnected';
      orbTitle.textContent = 'Disconnected';
      orbSubtitle.textContent = 'Tap to reconnect voice dispatch';
      break;
    default:
      cssClass = 'idle';
      orbTitle.textContent = 'Voice Assistant';
      orbSubtitle.textContent = detail || 'Ready';
      break;
  }
  voiceOrb.className = `voice-orb ${cssClass}`;
}

function appendTranscript(role, text, msgId, turnSeq) {
  if (!text || !text.trim()) return;
  const trimmed = text.trim();
  const now = Date.now();

  // Deduplicate identical message content from same speaker within 1500ms
  if (lastRenderedRole === role && lastRenderedText === trimmed && (now - lastRenderedTime < 1500)) {
    console.warn(`[TRANSCRIPT] Deduplicating identical message within 1500ms: "${trimmed}"`);
    return;
  }

  if (msgId && renderedMsgIds.has(msgId)) return;
  if (turnSeq !== undefined && turnSeq < clientTurnSeq) {
    console.warn(`[TRANSCRIPT] Discarding stale transcript from turn_seq ${turnSeq} (current: ${clientTurnSeq})`);
    return;
  }
  if (msgId) {
    renderedMsgIds.add(msgId);
  }

  lastRenderedRole = role;
  lastRenderedText = trimmed;
  lastRenderedTime = now;

  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${role}`;

  const speakerTag = document.createElement('span');
  speakerTag.className = 'speaker-tag';
  speakerTag.textContent = role === 'bot' ? 'Disha (Voice Concierge)' : 'You';

  const content = document.createElement('p');
  content.textContent = trimmed;

  bubble.appendChild(speakerTag);
  bubble.appendChild(content);

  transcriptBox.appendChild(bubble);
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function renderPoiCandidates(candidates) {
  if (!candidates || candidates.length === 0) {
    poiPillsSection.style.display = 'none';
    return;
  }

  // Clear previous Leaflet POI markers
  leafletPoiMarkers.forEach(m => {
    if (leafletMap) leafletMap.removeLayer(m);
  });
  leafletPoiMarkers = [];

  // Plot candidate markers if Leaflet is active
  if (leafletMap && typeof L !== 'undefined') {
    const featureGroup = L.featureGroup();
    if (leafletPickupMarker) {
      featureGroup.addLayer(leafletPickupMarker);
    }

    candidates.slice(0, 4).forEach((c, idx) => {
      const poiIcon = L.divIcon({
        className: 'custom-leaflet-poi',
        html: `
          <div style="background: #2563eb; color: #ffffff; width: 26px; height: 26px; border-radius: 50%; border: 2px solid #ffffff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">
            ${idx + 1}
          </div>
        `,
        iconSize: [26, 26],
        iconAnchor: [13, 13]
      });

      const m = L.marker([c.lat, c.lon], {
        icon: poiIcon,
        title: `${c.name} (${c.distance_km} km)`
      }).addTo(leafletMap);

      m.bindPopup(`
        <div style="color: #0f172a; font-family: sans-serif; font-size: 13px; line-height: 1.4;">
          <strong>${c.name}</strong><br>
          ${c.short_address || ''}<br>
          <span style="color: #2563eb; font-weight: 600;">${c.distance_km} km away</span>
        </div>
      `);

      m.on('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'SELECT_POI',
            lat: c.lat,
            lon: c.lon,
            name: c.name
          }));
        }
        poiPillsSection.style.display = 'none';
      });

      leafletPoiMarkers.push(m);
      featureGroup.addLayer(m);
    });

    try {
      leafletMap.fitBounds(featureGroup.getBounds().pad(0.15));
    } catch (e) {
      console.warn('fitBounds error:', e);
    }
  }

  // Render clickable venue chips
  poiPillsList.innerHTML = '';
  candidates.slice(0, 4).forEach((c, i) => {
    const card = document.createElement('button');
    card.className = 'venue-pill-chip';
    card.innerHTML = `
      <span class="venue-index">${i + 1}</span>
      <span class="venue-name">${c.name}</span>
      <span class="venue-distance">• ${c.distance_km.toFixed(1)} km</span>
    `;
    card.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'SELECT_POI',
          lat: c.lat,
          lon: c.lon,
          name: c.name
        }));
      }
      poiPillsSection.style.display = 'none';
    };
    poiPillsList.appendChild(card);
  });

  poiPillsSection.style.display = 'block';
}

function setTripCardState(state, data) {
  if (cardIdleView) cardIdleView.style.display = 'none';
  if (cardProposalView) cardProposalView.style.display = 'none';
  if (cardActiveView) cardActiveView.style.display = 'none';
  if (cardCancelledView) cardCancelledView.style.display = 'none';

  if (state === 'idle') {
    currentRideState = 'IDLE';
    activeRideData = null;
    clearTripPanelError();
    if (cardIdleView) cardIdleView.style.display = 'block';
  } else if (state === 'proposal') {
    currentRideState = 'PROPOSAL';
    if (cardProposalView) cardProposalView.style.display = 'block';
  } else if (state === 'active') {
    currentRideState = 'RIDE_ACTIVE';
    activeRideData = data;
    if (cardActiveView) cardActiveView.style.display = 'block';
  } else if (state === 'cancelled') {
    currentRideState = 'CANCELLED';
    activeRideData = null;
    if (cardCancelledView) cardCancelledView.style.display = 'block';
  }
}

function recordActivity() {
  lastUserActivityTime = Date.now();
}

function pauseDueToInactivity() {
  console.info('[INACTIVITY] Pausing audio capture after idle time');
  stopAudioCapture('GLOBAL_INACTIVITY');
  setOrbState(SESSION_STATES.PAUSED, 'Paused due to inactivity');
}

setInterval(() => {
  if (isRecording && Date.now() - lastUserActivityTime > 5 * 60 * 1000) {
    pauseDueToInactivity();
  }
}, 5000);

function renderRoutePreview(data) {
  console.log('Rendering Route Preview:', data);
  recordActivity();

  // Clear POI search markers from map
  leafletPoiMarkers.forEach(m => {
    if (leafletMap) leafletMap.removeLayer(m);
  });
  leafletPoiMarkers = [];

  // Update Trip Summary HUD
  if (cardOriginName) cardOriginName.textContent = `Current Location (${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E)`;
  if (cardDestName) cardDestName.textContent = data.destination || 'Destination';
  if (cardDistance) cardDistance.textContent = `${Number(data.distance_km).toFixed(1)} km`;
  if (cardDuration) cardDuration.textContent = `${data.duration_mins} mins`;
  if (cardFareAmount) cardFareAmount.textContent = `₹${Math.round(Number(data.fare_inr))}`;

  setTripCardState('proposal', data);
  poiPillsSection.style.display = 'none';

  // Leaflet OpenStreetMap Cartography
  if (leafletMap && typeof L !== 'undefined') {
    // Clear existing route polyline
    if (leafletRoutePolyline) {
      leafletMap.removeLayer(leafletRoutePolyline);
      leafletRoutePolyline = null;
    }
    // Clear existing destination pin
    if (leafletDestMarker) {
      leafletMap.removeLayer(leafletDestMarker);
      leafletDestMarker = null;
    }

    const featureGroup = L.featureGroup();
    if (leafletPickupMarker) {
      featureGroup.addLayer(leafletPickupMarker);
    }

    // Draw route polyline if geometry exists
    if (data.route_geometry && data.route_geometry.coordinates && data.route_geometry.coordinates.length > 0) {
      // GeoJSON coordinates are [lon, lat], Leaflet expects [lat, lon]
      const latLngs = data.route_geometry.coordinates.map(pt => [pt[1], pt[0]]);

      leafletRoutePolyline = L.polyline(latLngs, {
        color: '#2563eb', // Executive Primary Blue
        weight: 5,
        opacity: 0.9,
        lineJoin: 'round'
      }).addTo(leafletMap);

      featureGroup.addLayer(leafletRoutePolyline);
    }

    // Add Coral Red Destination Pin
    if (data.dest) {
      const destIcon = L.divIcon({
        className: 'custom-leaflet-dest',
        html: `
          <svg xmlns="http://www.w3.org/2000/svg" width="30" height="38" viewBox="0 0 32 42">
            <path d="M16 0C7.16 0 0 7.16 0 16c0 12 16 26 16 26s16-14 16-26C32 7.16 24.84 0 16 0z" fill="#ef4444"/>
            <circle cx="16" cy="16" r="6" fill="#ffffff"/>
          </svg>
        `,
        iconSize: [30, 38],
        iconAnchor: [15, 38]
      });

      leafletDestMarker = L.marker([data.dest.lat, data.dest.lon], {
        icon: destIcon,
        title: data.destination || 'Destination'
      }).addTo(leafletMap);

      featureGroup.addLayer(leafletDestMarker);
    }

    try {
      leafletMap.fitBounds(featureGroup.getBounds().pad(0.15));
    } catch (e) {
      console.warn('fitBounds error:', e);
    }
  }
}

function renderBookingConfirmed(data) {
  recordActivity();
  if (confirmedBookingId) {
    confirmedBookingId.textContent = data.booking_id || 'BOC-CONFIRMED';
  }
  if (confirmedDistance) {
    confirmedDistance.textContent = (data.distance_km != null) ? `${data.distance_km} km` : '-- km';
  }
  if (confirmedDuration) {
    confirmedDuration.textContent = (data.duration_mins != null) ? `${data.duration_mins} mins` : '-- mins';
  }
  if (confirmedFare) {
    confirmedFare.textContent = (data.fare_inr != null) ? `₹${data.fare_inr}` : '₹--';
  }

  setTripCardState('active', data);

  if (liveStatusLabel) {
    liveStatusLabel.textContent = 'Active Simulated Ride';
  }

  // Defect 4 fix: Cleanly terminate audio capture after booking is confirmed
  if (activeAudioSources.length > 0) {
    isBookingConfirmedPendingSpeech = true;
    console.info('[LIFECYCLE] Booking confirmed; waiting for confirmation speech to finish before closing mic');
  } else {
    stopAudioCapture('BOOKING_CONFIRMED_IMMEDIATE');
    setOrbState(SESSION_STATES.READY, 'Booking confirmed');
  }

  voiceOrb.style.boxShadow = '0 0 50px rgba(16, 185, 129, 0.8)';
  setTimeout(() => {
    voiceOrb.style.boxShadow = '';
  }, 2500);
}

function handleRideCancelled() {
  recordActivity();
  isBookingConfirmedPendingSpeech = false;
  if (leafletRoutePolyline && leafletMap) {
    leafletMap.removeLayer(leafletRoutePolyline);
    leafletRoutePolyline = null;
  }
  if (leafletDestMarker && leafletMap) {
    leafletMap.removeLayer(leafletDestMarker);
    leafletDestMarker = null;
  }
  leafletPoiMarkers.forEach(m => {
    if (leafletMap) leafletMap.removeLayer(m);
  });
  leafletPoiMarkers = [];

  setTripCardState('cancelled');
  poiPillsSection.style.display = 'none';

  if (liveStatusLabel) {
    liveStatusLabel.textContent = 'Voice Core Active';
  }

  if (leafletMap) {
    leafletMap.setView([userLocation.lat, userLocation.lon], 14);
  }
}

function sendTextMessage(text) {
  if (!text || !text.trim() || isSendingText) return;
  isSendingText = true;
  recordActivity();
  clearTripPanelError();
  clearInterimTranscript();

  clientTurnSeq++;
  const payload = {
    type: 'TEXT_INPUT',
    text: text.trim(),
    turn_seq: clientTurnSeq
  };

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  } else {
    connectWebSocket().then(() => {
      ws.send(JSON.stringify(payload));
    });
  }

  setTimeout(() => {
    isSendingText = false;
  }, 300);
}

function clearConversationHistory() {
  recordActivity();
  clearAllTimeouts();
  clearTripPanelError();
  clearInterimTranscript();
  interruptPlayback();
  clientTurnSeq++;
  lastRenderedRole = null;
  lastRenderedText = null;
  lastRenderedTime = 0;
  renderedMsgIds.clear();
  transcriptBox.innerHTML = '';

  if (currentRideState !== 'RIDE_ACTIVE') {
    setTripCardState('idle');
    if (leafletRoutePolyline && leafletMap) {
      leafletMap.removeLayer(leafletRoutePolyline);
      leafletRoutePolyline = null;
    }
    if (leafletDestMarker && leafletMap) {
      leafletMap.removeLayer(leafletDestMarker);
      leafletDestMarker = null;
    }
    leafletPoiMarkers.forEach(m => {
      if (leafletMap) leafletMap.removeLayer(m);
    });
    leafletPoiMarkers = [];
    poiPillsSection.style.display = 'none';
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'CLEAR_CONVERSATION',
      turn_seq: clientTurnSeq
    }));
  }

  if (isRecording) {
    setOrbState(SESSION_STATES.LISTENING);
  } else {
    setOrbState(SESSION_STATES.READY);
  }
  console.info(`[CLEAR] Conversation cleared (clientTurnSeq=${clientTurnSeq}, rideActive=${currentRideState === 'RIDE_ACTIVE'})`);
}

// ----------------------------------------------------------------------------- 8. Event Listeners
voiceOrb.addEventListener('click', async () => {
  recordActivity();
  clearAllTimeouts();

  // If in ERROR state, retry cleanly without duplicating the previous turn
  if (currentSessionState === SESSION_STATES.ERROR || orbState === SESSION_STATES.ERROR) {
    console.info('[LIFECYCLE] User tapped orb in ERROR state -> clearing error and retrying cleanly');
    clearTripPanelError();
    clearInterimTranscript();
    clientTurnSeq++;
    try {
      await connectWebSocket();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'START_VOICE_SESSION' }));
      }
      await startAudioCapture();
    } catch (e) {
      console.error('Failed to retry voice capture after error:', e);
    }
    return;
  }

  // If paused due to inactivity, resume capture
  if (currentSessionState === SESSION_STATES.PAUSED || voiceOrb.classList.contains('paused') || voiceOrb.classList.contains('paused_inactivity')) {
    try {
      await connectWebSocket();
      await startAudioCapture();
    } catch (e) {
      console.error('Failed to resume voice capture:', e);
    }
    return;
  }

  if (!isRecording) {
    try {
      await connectWebSocket();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'START_VOICE_SESSION' }));
      }
      await startAudioCapture();
    } catch (e) {
      console.error('Failed to start voice capture:', e);
    }
  } else {
    stopAudioCapture('USER_CLICK');
    setOrbState(SESSION_STATES.READY);
  }
});

btnConfirmRide.addEventListener('click', () => {
  recordActivity();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'CONFIRM_BOOKING' }));
  }
});

if (btnCancelRide) {
  btnCancelRide.addEventListener('click', () => {
    recordActivity();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'CANCEL_RIDE' }));
    }
  });
}

if (btnChangeDestination) {
  btnChangeDestination.addEventListener('click', () => {
    recordActivity();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'CHANGE_DESTINATION' }));
    }
    if (leafletRoutePolyline && leafletMap) {
      leafletMap.removeLayer(leafletRoutePolyline);
      leafletRoutePolyline = null;
    }
    if (leafletDestMarker && leafletMap) {
      leafletMap.removeLayer(leafletDestMarker);
      leafletDestMarker = null;
    }
    setTripCardState('idle');
  });
}

if (btnPlanNewTrip) {
  btnPlanNewTrip.addEventListener('click', () => {
    recordActivity();
    setTripCardState('idle');
  });
}

sendBtn.addEventListener('click', () => {
  const val = textInput.value;
  if (val && val.trim()) {
    sendTextMessage(val);
    textInput.value = '';
  }
});

textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const val = textInput.value;
    if (val && val.trim()) {
      sendTextMessage(val);
      textInput.value = '';
    }
  }
});

clearTranscriptBtn.addEventListener('click', () => {
  clearConversationHistory();
});

document.querySelectorAll('.lang-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    const lang = pill.dataset.lang;
    setLanguage(lang);
  });
});

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  loadMapEngine();
});
