/**
 * Bharat Online Cabs - Autonomous Voice Dispatch Client
 * Full-duplex Web Audio capture via AudioWorklet (16kHz PCM16) & streaming playback (22.05kHz PCM16).
 * Real Google Maps JavaScript API with responsive polyline, POI discovery, and honest dispatch telemetry.
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
let workletNode = null;
let micProcessor = null; // fallback
let isRecording = false;
let nextPlayTime = 0;
let activeAudioSources = [];
let userLocation = { lat: DEFAULT_LAT, lon: DEFAULT_LON };
let currentLanguage = 'en'; // 'en' | 'hi' | 'mr'

// Google Maps State
let googleMap = null;
let googlePickupMarker = null;
let googleDestMarker = null;
let googleRoutePolyline = null;
let googlePoiMarkers = [];
let googleMapsLoaded = false;
let googleMapsBrowserKey = '';

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
const cardOriginName = document.getElementById('cardOriginName');
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
const mapFallbackBanner = document.getElementById('mapFallbackBanner');

// ----------------------------------------------------------------------------- 1. Google Maps Initialization
window.gm_authFailure = function () {
  console.warn('Google Maps authentication failure. Check GOOGLE_MAPS_BROWSER_API_KEY restrictions.');
  if (mapFallbackBanner) {
    mapFallbackBanner.style.display = 'flex';
    mapFallbackBanner.querySelector('.banner-text span').innerHTML =
      'Google Maps API Key Authentication Failed. Check API restrictions in Google Cloud Console.';
  }
};

async function loadMapEngine() {
  try {
    const res = await fetch('/api/config');
    const config = await res.json();
    googleMapsBrowserKey = config.google_maps_browser_key || '';

    if (config.default_lat && config.default_lon) {
      userLocation.lat = config.default_lat;
      userLocation.lon = config.default_lon;
    }

    if (googleMapsBrowserKey) {
      loadGoogleMapsScript(googleMapsBrowserKey);
    } else {
      console.info('GOOGLE_MAPS_BROWSER_API_KEY not configured. Displaying setup guidance.');
      showMapConfigBanner();
      renderMapPlaceholder();
    }
  } catch (err) {
    console.error('Error fetching /api/config:', err);
    showMapConfigBanner();
    renderMapPlaceholder();
  }

  // Attempt browser GPS acquisition
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        userLocation.lat = pos.coords.latitude;
        userLocation.lon = pos.coords.longitude;
        if (gpsCoordsLabel) {
          gpsCoordsLabel.textContent = `GPS (${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E)`;
        }
        if (googleMap && googlePickupMarker) {
          const latLng = new google.maps.LatLng(userLocation.lat, userLocation.lon);
          googlePickupMarker.setPosition(latLng);
          googleMap.setCenter(latLng);
        }
      },
      () => {
        console.log('Using default Nagpur coordinates');
      },
      { timeout: 5000 }
    );
  }
}

function showMapConfigBanner() {
  if (mapFallbackBanner) {
    mapFallbackBanner.style.display = 'flex';
  }
}

function loadGoogleMapsScript(apiKey) {
  if (window.google && window.google.maps) {
    initGoogleMap();
    return;
  }

  const script = document.createElement('script');
  script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&libraries=geometry`;
  script.async = true;
  script.defer = true;
  script.onload = () => {
    googleMapsLoaded = true;
    initGoogleMap();
  };
  script.onerror = () => {
    console.error('Failed to load Google Maps script.');
    showMapConfigBanner();
    renderMapPlaceholder();
  };
  document.head.appendChild(script);
}

function initGoogleMap() {
  const mapElement = document.getElementById('map');
  if (!mapElement || !window.google || !window.google.maps) return;

  const centerLatLng = { lat: userLocation.lat, lng: userLocation.lon };
  googleMap = new google.maps.Map(mapElement, {
    center: centerLatLng,
    zoom: 14,
    disableDefaultUI: false,
    zoomControl: true,
    mapTypeControl: false,
    streetViewControl: false,
    fullscreenControl: false,
    styles: [
      { featureType: 'poi.business', stylers: [{ visibility: 'on' }] },
      { featureType: 'transit', stylers: [{ visibility: 'on' }] }
    ]
  });

  // Pickup marker (Radar pin)
  const pickupSvg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30">
      <circle cx="15" cy="15" r="12" fill="#10b981" fill-opacity="0.25"/>
      <circle cx="15" cy="15" r="7" fill="#10b981" stroke="#ffffff" stroke-width="2.5"/>
    </svg>
  `;

  googlePickupMarker = new google.maps.Marker({
    position: centerLatLng,
    map: googleMap,
    title: 'Pickup Location (Nagpur)',
    icon: {
      url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(pickupSvg),
      scaledSize: new google.maps.Size(30, 30),
      anchor: new google.maps.Point(15, 15)
    }
  });

  if (mapFallbackBanner) {
    mapFallbackBanner.style.display = 'none';
  }
}

function renderMapPlaceholder() {
  const mapElement = document.getElementById('map');
  if (!mapElement || googleMap) return;

  mapElement.innerHTML = `
    <div style="width: 100%; height: 100%; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); display: flex; flex-direction: column; align-items: center; justify-content: center; color: #94a3b8; text-align: center; padding: 24px;">
      <div style="width: 56px; height: 56px; border-radius: 50%; background: rgba(37, 99, 235, 0.15); border: 2px solid #2563eb; display: flex; align-items: center; justify-content: center; margin-bottom: 14px;">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2">
          <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon>
          <line x1="8" y1="2" x2="8" y2="18"></line>
          <line x1="16" y1="6" x2="16" y2="22"></line>
        </svg>
      </div>
      <h3 style="color: #f1f5f9; font-size: 1.05rem; margin-bottom: 6px;">Nagpur Mobility Grid Ready</h3>
      <p style="font-size: 0.82rem; max-width: 320px; line-height: 1.4;">Live coordinates: ${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E.<br>Provide <code>GOOGLE_MAPS_BROWSER_API_KEY</code> in <code>.env</code> to render vector cartography.</p>
    </div>
  `;
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

// ----------------------------------------------------------------------------- 4. Audio Ingestion via AudioWorklet
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

    // AudioWorklet Pipeline
    if (audioCaptureCtx.audioWorklet) {
      try {
        await audioCaptureCtx.audioWorklet.addModule('/static/pcm-recorder-worklet.js');
        workletNode = new AudioWorkletNode(audioCaptureCtx, 'pcm-recorder-processor');
        workletNode.port.onmessage = (e) => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };
        source.connect(workletNode);
        isRecording = true;
        setOrbState('listening');
        return;
      } catch (workletErr) {
        console.warn('AudioWorklet module failed, falling back to ScriptProcessor:', workletErr);
      }
    }

    // Fallback ScriptProcessor
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
  } catch (err) {
    console.error('Microphone access error:', err);
    setOrbState('error');
    orbTitle.textContent = 'Microphone Blocked';
    orbSubtitle.textContent = 'Please enable microphone access in your browser';
  }
}

function stopAudioCapture() {
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
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
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function renderPoiCandidates(candidates) {
  if (!candidates || candidates.length <= 1) {
    poiPillsSection.style.display = 'none';
    return;
  }

  // Clear previous Google Maps POI markers
  googlePoiMarkers.forEach(m => m.setMap(null));
  googlePoiMarkers = [];

  // Plot candidate markers if Google Maps is active
  if (googleMap && window.google) {
    const bounds = new google.maps.LatLngBounds();
    bounds.extend(new google.maps.LatLng(userLocation.lat, userLocation.lon));

    candidates.slice(0, 4).forEach((c, idx) => {
      const pos = new google.maps.LatLng(c.lat, c.lon);
      const m = new google.maps.Marker({
        position: pos,
        map: googleMap,
        title: `${c.name} (${c.distance_km} km)`,
        label: {
          text: String(idx + 1),
          color: '#ffffff',
          fontWeight: '700',
          fontSize: '12px'
        },
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 13,
          fillColor: '#2563eb',
          fillOpacity: 1,
          strokeColor: '#ffffff',
          strokeWeight: 2.5
        }
      });

      const infoWindow = new google.maps.InfoWindow({
        content: `<div style="color: #0f172a; font-family: sans-serif; font-size: 13px;"><strong>${c.name}</strong><br>${c.short_address || ''}<br><span style="color: #2563eb; font-weight: 600;">${c.distance_km} km away</span></div>`
      });

      m.addListener('click', () => {
        infoWindow.open(googleMap, m);
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

      googlePoiMarkers.push(m);
      bounds.extend(pos);
    });

    googleMap.fitBounds(bounds, { top: 60, bottom: 60, left: 60, right: 60 });
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

function renderRoutePreview(data) {
  console.log('Rendering Route Preview:', data);

  // Clear POI search markers from map
  googlePoiMarkers.forEach(m => m.setMap(null));
  googlePoiMarkers = [];

  // Update Trip Summary HUD
  if (cardOriginName) cardOriginName.textContent = `Current Location (${userLocation.lat.toFixed(4)}° N, ${userLocation.lon.toFixed(4)}° E)`;
  if (cardDestName) cardDestName.textContent = data.destination || 'Destination';
  if (cardDistance) cardDistance.textContent = `${Number(data.distance_km).toFixed(1)} km`;
  if (cardDuration) cardDuration.textContent = `${data.duration_mins} mins`;
  if (cardFareAmount) cardFareAmount.textContent = `₹${Math.round(Number(data.fare_inr))}`;

  rideCard.style.display = 'block';
  poiPillsSection.style.display = 'none';

  // Google Maps Cartography
  if (googleMap && window.google) {
    // Clear existing route polyline
    if (googleRoutePolyline) {
      googleRoutePolyline.setMap(null);
      googleRoutePolyline = null;
    }
    // Clear existing destination pin
    if (googleDestMarker) {
      googleDestMarker.setMap(null);
      googleDestMarker = null;
    }

    const bounds = new google.maps.LatLngBounds();
    bounds.extend(new google.maps.LatLng(userLocation.lat, userLocation.lon));

    // Draw route polyline if geometry exists
    if (data.route_geometry && data.route_geometry.coordinates) {
      const path = data.route_geometry.coordinates.map(pt => ({
        lat: pt[1],
        lng: pt[0]
      }));

      googleRoutePolyline = new google.maps.Polyline({
        path: path,
        geodesic: true,
        strokeColor: '#2563eb', // Executive Primary Blue
        strokeOpacity: 0.9,
        strokeWeight: 5
      });
      googleRoutePolyline.setMap(googleMap);

      path.forEach(pt => bounds.extend(new google.maps.LatLng(pt.lat, pt.lng)));
    }

    // Add Coral Red Destination Pin
    if (data.dest) {
      const destPos = new google.maps.LatLng(data.dest.lat, data.dest.lon);
      bounds.extend(destPos);

      const destSvg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="42" viewBox="0 0 32 42">
          <path d="M16 0C7.16 0 0 7.16 0 16c0 12 16 26 16 26s16-14 16-26C32 7.16 24.84 0 16 0z" fill="#ef4444"/>
          <circle cx="16" cy="16" r="6" fill="#ffffff"/>
        </svg>
      `;

      googleDestMarker = new google.maps.Marker({
        position: destPos,
        map: googleMap,
        title: data.destination,
        icon: {
          url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(destSvg),
          scaledSize: new google.maps.Size(32, 42),
          anchor: new google.maps.Point(16, 42)
        }
      });
    }

    googleMap.fitBounds(bounds, { top: 50, bottom: 50, left: 50, right: 50 });
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
  if (googleRoutePolyline) {
    googleRoutePolyline.setMap(null);
    googleRoutePolyline = null;
  }
  if (googleDestMarker) {
    googleDestMarker.setMap(null);
    googleDestMarker = null;
  }
  googlePoiMarkers.forEach(m => m.setMap(null));
  googlePoiMarkers = [];

  rideCard.style.display = 'none';
  bookingCard.style.display = 'none';
  poiPillsSection.style.display = 'none';

  if (liveStatusLabel) {
    liveStatusLabel.textContent = 'Voice Core Active';
  }
  appendTranscript('bot', 'Your ride has been cancelled. Where would you like to go instead?');

  if (googleMap) {
    googleMap.setCenter(new google.maps.LatLng(userLocation.lat, userLocation.lon));
    googleMap.setZoom(14);
  }
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
  loadMapEngine();
});
