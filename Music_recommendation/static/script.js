// Global UI elements
const video = document.getElementById("video");
const status = document.getElementById("status");
const emotionText = document.getElementById("emotionText");
const confidenceText = document.getElementById("confidenceText");
const langButtons = document.getElementById("langButtons");
const songsArea = document.getElementById("songsArea");
const avatarShell = document.getElementById("avatarShell");
const avatarEmotion = document.getElementById("avatarEmotion");
const moodChartCtx = document.getElementById("moodChart").getContext("2d");
const autoDetectBtn = document.getElementById("autoDetectBtn");
const autoStateSpan = document.getElementById("autoState");
const gestureStatus = document.getElementById("gestureStatus");

// Spotify UI and Variables
const spotifyAuthStatus = document.getElementById("spotifyAuthStatus");
const spotifyLoginLink = document.getElementById("spotifyLoginLink");
const spotifyPlayerInterface = document.getElementById("spotifyPlayerInterface");
const playbackStatus = document.getElementById("playbackStatus");
const trackNameEl = document.getElementById("trackName");
const artistNameEl = document.getElementById("artistName");
const trackProgressEl = document.getElementById("trackProgress");
const trackBarEl = document.getElementById("trackBar");

// Player Controls
const playPauseBtn = document.getElementById("playPauseBtn");

// Training elements (still included for completeness)
const videoTrain = document.getElementById("videoTrain");
const captureTrainBtn = document.getElementById("captureTrainBtn");
const trainLabel = document.getElementById("trainLabel");
const trainStatus = document.getElementById("trainStatus");
const datasetBtn = document.getElementById("datasetBtn");
const resetBtn = document.getElementById("resetBtn");
const themeToggle = document.getElementById("themeToggle");


let liveStream = null, trainStream = null;
let autoDetect = false;
let autoInterval = null;
let lastExecutedGesture = "NONE"; 
let gestureCooldownActive = false; 

let spotifyPlayer = null;
let spotifyAccessToken = null;
let spotifyDeviceID = null;

// ---------- Initialization and Setup ----------

// 1. Camera Initialization
async function startCamera(videoEl) {
    try {
        const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        videoEl.srcObject = s;
        await videoEl.play();
        status.textContent = "System Status: Camera Active";
        return s;
    } catch (e) {
        console.error("Camera start failed", e);
        status.textContent = "System Status: Camera Error";
        return null;
    }
}

function captureFromVideo(videoEl) {
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg");
}

(async function initCams(){
    const v1 = document.getElementById("video");
    const v2 = document.getElementById("videoTrain");
    if (v1) await startCamera(v1);
    if (v2) await startCamera(v2);
})();

// 2. Spotify Login Check
spotifyLoginLink.onclick = (e) => {
    e.preventDefault();
    window.location.href = "/login";
};

async function checkLoginStatus() {
    const res = await fetch("/is_logged_in");
    const data = await res.json();
    
    if (data.logged_in) {
        spotifyAccessToken = data.access_token;
        spotifyAuthStatus.textContent = "Spotify: Connected";
        spotifyAuthStatus.style.borderColor = '#a5d8ff'; 
        
        window.onSpotifyWebPlaybackSDKReady = initializeSpotifyPlayer;
        playbackStatus.textContent = "Loading Spotify Player...";
        spotifyPlayerInterface.classList.remove("hidden");
    } else {
        spotifyAuthStatus.textContent = "Spotify: Logged Out";
        spotifyAuthStatus.style.borderColor = '#ff6b6b';
        spotifyPlayerInterface.classList.add("hidden");
    }
}
checkLoginStatus();

// 3. Spotify Player Setup (Fixed State Listener)
function initializeSpotifyPlayer() {
    spotifyPlayer = new Spotify.Player({
        name: 'LoFi MoodPlayer (Browser)',
        getOAuthToken: cb => { cb(spotifyAccessToken); },
        volume: 0.5
    });

    spotifyPlayer.addListener('ready', ({ device_id }) => {
        spotifyDeviceID = device_id;
        playbackStatus.textContent = `Player Ready. Device ID: ${device_id.substring(0, 8)}...`;
        
        fetch("/set_device", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device_id: device_id })
        });
    });

    spotifyPlayer.addListener('player_state_changed', state => {
        if (!state || !state.track_window.current_track) return;
        
        const track = state.track_window.current_track;
        const duration = track.duration_ms;
        const position = state.position;
        const isPlaying = !state.paused;

        trackNameEl.textContent = track.name;
        artistNameEl.textContent = track.artists[0].name;

        // Display correct button icon
        playPauseBtn.innerHTML = isPlaying ? "⏸️" : "▶️";

        playbackStatus.innerHTML = isPlaying 
            ? `**Playing** (${Math.floor(position/1000)}s / ${Math.floor(duration/1000)}s)`
            : `**Paused**`;
        
        const progressPercent = (position / duration) * 100;
        trackProgressEl.style.width = `${progressPercent}%`;
        trackBarEl.setAttribute('data-duration', duration);
    });

    spotifyPlayer.connect();
}

// 4. Player Control Button Fix
playPauseBtn.addEventListener('click', async () => {
    if (!spotifyPlayer || !spotifyAccessToken) {
        alert("Please log in to Spotify.");
        return;
    }
    const state = await spotifyPlayer.getCurrentState();
    // Determine command based on current state
    const command = state && !state.paused ? 'PLAY_PAUSE_PAUSE' : 'PLAY_PAUSE_PLAY';
    await executeGestureCommand(command);
});


// 5. Seek Bar Functionality Fix
trackBarEl.addEventListener('click', (e) => {
    if (!spotifyPlayer || !spotifyDeviceID || !trackBarEl.getAttribute('data-duration')) {
        alert("Spotify player is not active or track information is missing.");
        return;
    }

    const duration = parseInt(trackBarEl.getAttribute('data-duration'));
    const rect = trackBarEl.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;

    const seekPositionMs = Math.round((clickX / width) * duration);
    
    // Use the SDK's built-in seek method
    spotifyPlayer.seek(seekPositionMs).then(() => {
        playbackStatus.textContent = `Seeking to ${Math.floor(seekPositionMs / 1000)}s...`;
    });
});

// 6. Chart Setup
const MAX_POINTS = 60;
const labels = Array(MAX_POINTS).fill("");
const dataPoints = Array(MAX_POINTS).fill(null);

const moodChart = new Chart(moodChartCtx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            label: 'Confidence (%)',
            data: dataPoints,
            tension: 0.35, borderWidth: 2.5, pointRadius: 2,
            borderColor: '#a5d8ff',
            backgroundColor: 'rgba(165, 216, 255, 0.1)',
            fill: true,
        }]
    },
    options: {
        responsive: true, animation: { duration: 400 },
        scales: { x: { display: false }, y: { min: 0, max: 100, ticks: { stepSize: 20, color: '#8892a0' }, grid: { color: 'rgba(255,255,255,0.03)' } } },
        plugins: { legend: { display: false } }
    }
});

function pushPoint(value) {
    labels.push('');
    labels.shift();
    dataPoints.push(value);
    dataPoints.shift();
    moodChart.update();
}

// 7. Utility Functions and Handlers

// Avatar Update
function setAvatar(emotion, confidence) {
    avatarShell.classList.remove("happy","sad","angry","neutral");
    const e = emotion ? emotion.toLowerCase() : "neutral";
    if (e.startsWith("hap")) avatarShell.classList.add("happy");
    else if (e.startsWith("sad")) avatarShell.classList.add("sad");
    else if (e.startsWith("ang")) avatarShell.classList.add("angry");
    else avatarShell.classList.add("neutral");

    avatarEmotion.textContent = `${emotion} • ${confidence}%`;
}

// Theme Toggle
themeToggle.addEventListener("click", () => {
    const isLight = document.documentElement.classList.toggle("light");
    localStorage.setItem("theme", isLight ? "light" : "dark");
});


// ---------- Core Detection and Control Logic ----------

// Auto Detect Toggle
autoDetectBtn.addEventListener("click", () => {
    autoDetect = !autoDetect;
    autoStateSpan.textContent = autoDetect ? "On" : "Off";
    
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent');
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg');

    if (autoDetect) {
        autoDetectBtn.style.backgroundColor = accent;
        autoDetectBtn.style.color = bg;
        
        autoInterval = setInterval(()=> {
            if (video && video.readyState >= 2) captureAndPredict(false);
        }, 600);
        
    } else {
        autoDetectBtn.style.backgroundColor = '';
        autoDetectBtn.style.color = '';
        clearInterval(autoInterval);
        autoInterval = null;
        status.textContent = "System Status: Idle";
        gestureStatus.textContent = "Hand Gesture: **None**";
        lastExecutedGesture = "NONE"; 
        gestureCooldownActive = false;
    }
});

// Gesture Execution (Simplified for Play/Pause)
async function executeGestureCommand(command) {
    const commandMap = {
        "PLAY_PAUSE_PAUSE": "Pause (✊)", 
        "PLAY_PAUSE_PLAY": "Play (✋)"
    };

    if (!commandMap[command]) return;

    gestureStatus.textContent = `Executing: ${commandMap[command]}...`;

    try {
        const res = await fetch("/gesture_command", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: command })
        });
        const j = await res.json();
        
        if (j.ok) {
            gestureStatus.textContent = `✅ Executed: ${j.status}`;
        } else {
            gestureStatus.textContent = `⚠️ Error: ${j.error || "failed"}`;
        }
    } catch (e) {
        gestureStatus.textContent = `Network Error: ${e}`;
    }
}

// Main Prediction Loop (Real-Time, Handles Gesture Automation)
async function captureAndPredict(showUI=true) {
    if (!video) return;
    
    const dataUrl = captureFromVideo(video);

    try {
        const res = await fetch("/predict", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl })
        });
        const j = await res.json();
        
        if (!j.ok) {
            status.textContent = "Detection Error: " + (j.error || "unknown");
            return null;
        }

        const emotion = j.prediction;
        const conf = Math.round((j.confidence || 0) * 100);
        const gesture = j.gesture; 

        // 1. Update Real-Time UI
        status.textContent = "System Status: Continuous Detection";
        emotionText.textContent = `Detected Mood: ${emotion}`;
        confidenceText.textContent = `Confidence: ${conf}%`;
        langButtons.setAttribute("data-emotion", emotion);
        
        setAvatar(emotion, conf);
        pushPoint(conf); 

        // 2. Handle Gesture (Automated Execution with Cooldown)
        if (gesture !== "NONE") {
            const displayGesture = gesture === 'PLAY_PAUSE_PLAY' ? '✋ Play' : '✊ Pause';
            gestureStatus.textContent = `Hand Gesture: **${displayGesture}**`;

            if (gesture !== lastExecutedGesture && !gestureCooldownActive) {
                
                gestureCooldownActive = true;
                lastExecutedGesture = gesture;

                await executeGestureCommand(gesture);
                
                setTimeout(() => {
                    gestureCooldownActive = false; 
                    lastExecutedGesture = "NONE"; 
                }, 1500); // 1.5s cooldown
            }
        } else {
            if (!gestureCooldownActive) {
                gestureStatus.textContent = "Hand Gesture: **None**";
            }
        }

        return {emotion, conf, gesture};
    } catch (e) {
        if (showUI) status.textContent = "Detection failed: " + e;
        return null;
    }
}

// ---------- Music and Training Handlers ----------

// Play Song (Initiated by clicking a song in the list)
async function playSong(uri, trackName) {
    if (!spotifyAccessToken || !spotifyDeviceID) {
        alert("Please log in with Spotify and ensure the player is connected (Premium needed).");
        return;
    }
    
    playbackStatus.textContent = `Attempting to play ${trackName}...`;

    try {
        await fetch("/play_song", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ uri: uri })
        });
        // State update handled by SDK listener
    } catch (e) {
        playbackStatus.textContent = `Playback Request Failed: ${e}`;
    }
}


// Language Button Handler
langButtons.addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".lang-btn");
    if (!btn) return;
    
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const lang = btn.getAttribute("data-lang");
    const emotion = langButtons.getAttribute("data-emotion");
    if (!emotion || emotion === 'Detected Mood: -') return alert("No emotion detected yet.");

    songsArea.innerHTML = `<div class="small muted">Fetching ${lang} songs for ${emotion}...</div>`;

    try {
        const res = await fetch("/songs", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ emotion: emotion, language: lang })
        });
        const j = await res.json();
        if (!j.ok) {
            songsArea.innerHTML = `<div class="small">Error fetching songs: ${j.error || "unknown"}</div>`;
            return;
        }

        const songs = j.songs || [];
        songsArea.innerHTML = songs.length === 0 ? `<div class="small">No songs found.</div>` : "";
        
        songs.forEach(track => {
            const b = document.createElement("button");
            b.className = "song-btn";
            b.innerHTML = `<div>${track.name}</div><div class="artist">${track.artist}</div>`;
            b.onclick = () => playSong(track.uri, track.name);
            songsArea.appendChild(b);
        });
    } catch (e) {
        songsArea.innerHTML = `<div class="small">Request failed: ${e}</div>`;
    }
});

// Train Capture Handler
captureTrainBtn.addEventListener("click", async () => {
    const label = trainLabel.value;
    trainStatus.textContent = "Capturing...";
    const dataUrl = captureFromVideo(videoTrain);

    try {
        const res = await fetch("/train", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl, label: label })
        });
        const j = await res.json();
        trainStatus.textContent = j.ok ? 
            `Saved (${label}) — total samples: ${j.count}` : 
            "Train error: " + (j.error || "unknown");
    } catch (e) {
        trainStatus.textContent = "Train failed: " + e;
    }
});

// Dataset and Reset Handlers
datasetBtn.onclick = async () => {
    try {
        const res = await fetch("/dataset");
        const j = await res.json();
        alert(`Total samples: ${j.count}\n` + Object.entries(j.by_label || {}).map(kv => `${kv[0]}: ${kv[1]}`).join("\n"));
    } catch (e) {
        alert("Dataset fetch failed: " + e);
    }
};

resetBtn.onclick = async () => {
    if (!confirm("Clear dataset.json?")) return;
    await fetch("/reset", { method: "POST" }).then(() => alert("Dataset cleared")).catch(e => alert("Reset failed: " + e));
};

// Seed chart with initial zeros
(function seedChart(){
    for (let i=0;i<8;i++) pushPoint(0);
})();