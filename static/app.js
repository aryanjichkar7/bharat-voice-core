/**
 * Bharat Online Cabs - Executive Voice Dispatch Client
 * Full-duplex Web Audio capture (16kHz PCM16) & streaming playback (22.05kHz PCM16).
 */

// Configuration & Default Coordinates (Nagpur, MH)
const DEFAULT_LAT = 21.1458;
const DEFAULT_LON = 79.0882;
const IN_SAMPLE_RATE = 16000;
const OUT_SAMPLE_RATE = 22050;

// Application State
let ws = null;
let audioCaptureCtx = null;
let audioPlaybackCtx = null;
let micStream = null;
let micProcessor = null;
let isRecording = false;
let nextPlayTime = 0;
let activeAudioSources = [];
let userLocation = { lat: DEFAULT_LAT, lon: DEFAULT_LON };
let currentLanguage = 'en'; // default: 'en' | 'hi' | 'mr'

let map = null;
let userMarker = null;
let destMarker = null;
let routeGeoJsonLayer = null;
let poiMarkers = [];

// DOM Elements
const voiceOrb = document.getElementById('voiceOrb');
const orbGlow = document.getElementById('orbGlow');
const orbTitle = document.getElementById('orbStateTitle');
const orbSubtitle = document.getElementById('orbStateSubtitle');

const liveStatusBadge = document.getElementById('liveStatusBadge');
const liveStatusLabel = document.getElementById('liveStatusLabel');
const gpsCoordsLabel = document.getElementById('gpsCoordsLabel');

const transcriptBox = document.getElementById('transcriptBox');
const textInput = document.getElementById('textInput');
const sendBtn = document.getElementById('sendBtn');
const clearTranscriptBtn = document.getElementById('clearTranscriptBtn');

const poiPillsSection = document.getElementById('poiPillsSection');
const poiPillsList = document.getElementById('poiPillsList');

const rideCard = document.getElementById('rideCard');
const cardDestName = document.getElementById('cardDestName');
const cardFareAmount = document.getElementById('cardFareAmount');
const cardDistance = document.getElementById('cardDistance');
const cardDuration = document.getElementById('cardDuration');
const btnConfirmRide = document.getElementById('btnConfirmRide');

const bookingCard = document.getElementById('bookingCard');
const confirmedBookingId = document.getElementById('confirmedBookingId');
const bookingOtpBadge = document.getElementById('bookingOtpBadge');
const driverName = document.getElementById('driverName');
const driverVehicle = document.getElementById('driverVehicle');
const driverEtaPill = document.getElementById('driverEtaPill');
const btnCancelRide = document.getElementById('btnCancelRide');

// ----------------------------------------------------------------------------- 1. Map Initialization
function initMap() {
  map = L.map('map', {
    zoomControl: false,
    attributionControl: false
  }).setView([userLocation.lat, userLocation.lon], 14);

  // Bulletproof, keyless OpenStreetMap tile layer
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  // Pickup Pin: Green pulsing circular radar
  const pickupIcon = L.divIcon({
    className: 'custom-pickup-marker',
    html: `
      <div class="radar-pin green">
        <div class="radar-ring"></div>
        <div class="radar-core"></div>
      </div>
    `,
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });

  userMarker = L.marker([userLocation.lat, userLocation.lon], { icon: pickupIcon }).addTo(map);
  userMarker.bindPopup('<b>Current Pickup Location</b><br>Nagpur, Maharashtra').openPopup();

  // Try Browser GPS
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        userLocation.lat = pos.coords.latitude;
        userLocation.lon = pos.coords.longitude;
        userMarker.setLatLng([userLocation.lat, userLocation.lon]);
        map.setView([userLocation.lat, userLocation.lon], 15);
        if (gpsCoordsLabel) {
          gpsCoordsLabel.textContent = `GPS (${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E)`;
        }
      },
      () => {
        console.log('Using default Nagpur coordinates');
      },
      { timeout: 5000 }
    );
  }
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

    setOrbState('connecting');
    orbTitle.textContent = 'Connecting...';
    orbSubtitle.textContent = 'Initializing secure voice session';

    ws = new WebSocket(getWsUrl());
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus(true);

      // Send CLIENT_INIT frame
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
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      setConnectionStatus(false);
      setOrbState('error');
      orbTitle.textContent = 'Connection Error';
      orbSubtitle.textContent = 'Could not reach voice dispatch';
      reject(err);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnectionStatus(false);
      stopAudioCapture();
      setOrbState('idle');
      orbTitle.textContent = 'Tap to Speak';
      orbSubtitle.textContent = 'Autonomous Voice Dispatch • Nagpur';
    };
  });
}

// ----------------------------------------------------------------------------- 4. Audio Ingestion & Resampling
async function startAudioCapture() {
  try {
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
    const bufferSize = 4096;
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

    setOrbState('listening');
    orbTitle.textContent = 'Listening...';
    orbSubtitle.textContent = `Speak in ${getLangName(currentLanguage)}`;
  } catch (err) {
    console.error('Microphone error:', err);
    setOrbState('error');
    orbTitle.textContent = 'Microphone Blocked';
    orbSubtitle.textContent = 'Please enable microphone access in your browser';
  }
}

function stopAudioCapture() {
  if (micProcessor) {
    micProcessor.disconnect();
    micProcessor = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  if (audioCaptureCtx && audioCaptureCtx.state !== 'closed') {
    audioCaptureCtx.close();
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
  if (!audioPlaybackCtx) {
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    audioPlaybackCtx = new AudioCtxClass({ sampleRate: OUT_SAMPLE_RATE });
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

  sourceNode.onended = () => {
    const idx = activeAudioSources.indexOf(sourceNode);
    if (idx > -1) activeAudioSources.splice(idx, 1);
    if (activeAudioSources.length === 0 && isRecording) {
      setOrbState('listening');
    }
  };

  setOrbState('speaking');
}

function interruptPlayback() {
  activeAudioSources.forEach(src => {
    try { src.stop(); } catch (e) {}
  });
  activeAudioSources = [];
  if (audioPlaybackCtx) {
    nextPlayTime = audioPlaybackCtx.currentTime;
  }
  if (isRecording) {
    setOrbState('listening');
  }
}

// ----------------------------------------------------------------------------- 6. Telemetry & Frame Handlers
function handleServerTextMessage(text) {
  try {
    const data = JSON.parse(text);
    console.log('Telemetry Frame:', data);

    switch (data.type) {
      case 'CONNECTION_STATUS':
        if (data.status === 'CONNECTED') {
          setConnectionStatus(true);
        }
        break;

      case 'SESSION_READY':
        setConnectionStatus(true);
        setOrbState('listening');
        orbTitle.textContent = 'Listening...';
        orbSubtitle.textContent = `Speak in ${getLangName(currentLanguage)}`;
        break;

      case 'LANGUAGE_CHANGED':
        if (data.language) {
          currentLanguage = data.language;
          updateLanguagePillsUI(data.language);
        }
        break;

      case 'USER_TRANSCRIPT':
        if (textInput) {
          textInput.value = data.text;
        }
        appendTranscript('user', data.text);
        break;

      case 'BOT_RESPONSE':
        appendTranscript('bot', data.text);
        break;

      case 'POI_CANDIDATES':
        renderPoiCandidates(data.candidates || []);
        break;

      case 'MAP_ROUTE_PREVIEW':
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
        console.error('Server error:', data.error);
        appendTranscript('bot', `Error: ${data.error}`);
        break;

      default:
        break;
    }
  } catch (err) {
    console.error('Non-JSON text frame:', text);
  }
}

// ----------------------------------------------------------------------------- 7. UI Rendering & Interactions
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
  currentLanguage = lang;
  updateLanguagePillsUI(lang);
  const initialBubble = document.getElementById('initialGreetingText');
  if (initialBubble) {
    const greetings = {
      'en': 'Hello! Welcome to Bharat Online Cabs. Where would you like to go today?',
      'hi': 'नमस्ते! भारत ऑनलाइन कैब्स में आपका स्वागत है। आप कहाँ जाना चाहेंगे?',
      'mr': 'नमस्कार! भारत ऑनलाईन कॅब्समध्ये आपले स्वागत आहे. आपल्याला कुठे जायचे आहे?'
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

function setOrbState(state) {
  voiceOrb.className = `voice-orb ${state}`;
  if (state === 'listening') {
    orbTitle.textContent = 'Listening...';
    orbSubtitle.textContent = `Speak in ${getLangName(currentLanguage)}`;
  } else if (state === 'speaking') {
    orbTitle.textContent = 'Disha Speaking...';
    orbSubtitle.textContent = 'Voice Concierge Response';
  } else if (state === 'idle') {
    orbTitle.textContent = 'Tap to Speak';
    orbSubtitle.textContent = 'Autonomous Voice Dispatch • Nagpur';
  }
}

function appendTranscript(role, text) {
  if (!text || !text.trim()) return;

  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${role}`;

  const speakerTag = document.createElement('span');
  speakerTag.className = 'speaker-tag';
  speakerTag.textContent = role === 'bot' ? 'Disha (Voice Concierge)' : 'You';

  const content = document.createElement('p');
  content.textContent = text.trim();

  bubble.appendChild(speakerTag);
  bubble.appendChild(content);

  transcriptBox.appendChild(bubble);
  const scrollBody = document.querySelector('.dock-body');
  if (scrollBody) {
    scrollBody.scrollTop = scrollBody.scrollHeight;
  }
}

function renderPoiCandidates(candidates) {
  if (!candidates || candidates.length <= 1) {
    poiPillsSection.style.display = 'none';
    return;
  }

  // Clear previous map POI markers
  poiMarkers.forEach(m => map.removeLayer(m));
  poiMarkers = [];

  // Plot Top 3 closest markers on the map
  const topCandidates = candidates.slice(0, 3);
  const bounds = L.latLngBounds([userLocation.lat, userLocation.lon]);

  topCandidates.forEach((c, idx) => {
    const badgeIcon = L.divIcon({
      className: 'custom-poi-marker',
      html: `<div class="poi-badge">${idx + 1}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    const m = L.marker([c.lat, c.lon], { icon: badgeIcon }).addTo(map);
    m.bindPopup(`<b>${c.name}</b><br>${c.short_address || ''}<br><b>${c.distance_km} km away</b>`);
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
    poiMarkers.push(m);
    bounds.extend([c.lat, c.lon]);
  });

  if (topCandidates.length > 0) {
    map.fitBounds(bounds, { padding: [60, 60], maxZoom: 16 });
  }

  // Render chips in console
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

function renderRoutePreview(data) {
  console.log('Rendering Route Preview:', data);

  // Clear POI search markers from map
  poiMarkers.forEach(m => map.removeLayer(m));
  poiMarkers = [];

  // Update Trip Summary HUD
  cardDestName.textContent = data.destination || 'Destination';
  cardDistance.textContent = `${Number(data.distance_km).toFixed(1)} km`;
  cardDuration.textContent = `${data.duration_mins} mins`;
  cardFareAmount.textContent = `₹${Number(data.fare_inr).toFixed(2)}`;

  rideCard.style.display = 'block';
  poiPillsSection.style.display = 'none';

  // Clear existing map layers
  if (routeGeoJsonLayer) {
    map.removeLayer(routeGeoJsonLayer);
  }
  if (destMarker) {
    map.removeLayer(destMarker);
  }

  // Draw Primary Blue Road Polyline (#2563eb, weight: 5)
  if (data.route_geometry) {
    routeGeoJsonLayer = L.geoJSON(data.route_geometry, {
      style: {
        color: '#2563eb',
        weight: 5,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round'
      }
    }).addTo(map);

    map.fitBounds(routeGeoJsonLayer.getBounds(), { padding: [40, 40] });
  }

  // Add Custom Coral Red Drop Pin with destination label
  if (data.dest) {
    const destIcon = L.divIcon({
      className: 'custom-dest-marker',
      html: `
        <div class="drop-pin red">
          <div class="drop-pin-badge">${data.destination || 'Destination'}</div>
          <svg width="28" height="34" viewBox="0 0 24 24" fill="#ef4444" stroke="#ffffff" stroke-width="2" style="filter: drop-shadow(0 3px 6px rgba(0,0,0,0.25));">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
            <circle cx="12" cy="9" r="2.5" fill="#ffffff"/>
          </svg>
        </div>
      `,
      iconSize: [32, 44],
      iconAnchor: [16, 34]
    });

    destMarker = L.marker([data.dest.lat, data.dest.lon], { icon: destIcon }).addTo(map);
    destMarker.bindPopup(`<b>${data.destination}</b><br>Fare: ₹${data.fare_inr}`).openPopup();
  }
}

function renderBookingConfirmed(data) {
  confirmedBookingId.textContent = data.booking_id || 'BOC-CONFIRMED';
  if (bookingOtpBadge) {
    bookingOtpBadge.textContent = `OTP: ${data.otp || '4821'}`;
  }
  if (driverName && data.driver?.name) {
    driverName.textContent = data.driver.name;
  }
  if (driverVehicle && data.driver?.car) {
    driverVehicle.textContent = `${data.driver.car} • ${data.driver.plate || 'MH 31 EQ 8421'}`;
  }
  if (driverEtaPill && data.driver?.eta_mins) {
    driverEtaPill.textContent = `${data.driver.eta_mins} mins away`;
  }

  bookingCard.style.display = 'block';
  rideCard.style.display = 'none';

  if (liveStatusLabel) {
    liveStatusLabel.textContent = 'Active Ride Monitoring';
  }

  voiceOrb.style.boxShadow = '0 0 50px rgba(16, 185, 129, 0.8)';
  setTimeout(() => {
    voiceOrb.style.boxShadow = '';
  }, 2500);
}

function handleRideCancelled() {
  if (routeGeoJsonLayer) {
    map.removeLayer(routeGeoJsonLayer);
    routeGeoJsonLayer = null;
  }
  if (destMarker) {
    map.removeLayer(destMarker);
    destMarker = null;
  }
  poiMarkers.forEach(m => map.removeLayer(m));
  poiMarkers = [];

  rideCard.style.display = 'none';
  bookingCard.style.display = 'none';
  poiPillsSection.style.display = 'none';

  if (liveStatusLabel) {
    liveStatusLabel.textContent = 'Voice Core Active';
  }
  appendTranscript('bot', 'Your ride has been cancelled. Where would you like to go instead?');
  map.setView([userLocation.lat, userLocation.lon], 14);
}

function sendTextMessage(text) {
  if (!text || !text.trim()) return;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'TEXT_INPUT',
      text: text.trim()
    }));
  } else {
    connectWebSocket().then(() => {
      ws.send(JSON.stringify({
        type: 'TEXT_INPUT',
        text: text.trim()
      }));
    });
  }
}

// ----------------------------------------------------------------------------- 8. Event Listeners
voiceOrb.addEventListener('click', async () => {
  if (!isRecording) {
    try {
      await connectWebSocket();
      await startAudioCapture();
    } catch (e) {
      console.error(e);
    }
  } else {
    stopAudioCapture();
    if (ws) ws.close();
  }
});

btnConfirmRide.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'CONFIRM_BOOKING' }));
  }
});

if (btnCancelRide) {
  btnCancelRide.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'CANCEL_RIDE' }));
    }
  });
}

sendBtn.addEventListener('click', () => {
  const val = textInput.value;
  if (val.trim()) {
    sendTextMessage(val);
    textInput.value = '';
  }
});

textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    sendBtn.click();
  }
});

clearTranscriptBtn.addEventListener('click', () => {
  transcriptBox.innerHTML = '';
});

document.querySelectorAll('.lang-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    const lang = pill.dataset.lang;
    setLanguage(lang);
  });
});

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  initMap();
});
