"""Module: call_component
Layer: Presentation / UI
Purpose: Full-duplex real-time streaming Voice Call Web Audio component with live interim transcription, waveform visualizer, and instant interruption (barge-in).
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Real-Time Voice Assistant</title>
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  body {
    background: transparent;
    color: #f3f4f6;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .call-card {
    width: 100%;
    max-width: 600px;
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
  }
  .avatar-container {
    position: relative;
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    transition: all 0.3s ease;
  }
  .avatar-container.active {
    box-shadow: 0 0 35px rgba(34, 197, 94, 0.7);
    background: linear-gradient(135deg, #10b981, #059669);
    animation: pulse 1.4s infinite;
  }
  .avatar-container.speaking {
    box-shadow: 0 0 35px rgba(168, 85, 247, 0.85);
    background: linear-gradient(135deg, #9333ea, #c084fc);
    animation: pulse-speak 1.2s infinite;
  }
  .avatar-container.interrupted {
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.8);
    background: linear-gradient(135deg, #ef4444, #f97316);
  }
  @keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 16px rgba(16, 185, 129, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
  }
  @keyframes pulse-speak {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.7); }
    70% { transform: scale(1.06); box-shadow: 0 0 0 18px rgba(168, 85, 247, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(168, 85, 247, 0); }
  }
  .avatar-icon {
    font-size: 42px;
  }
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 18px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 13px;
    font-weight: 500;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #9ca3af;
  }
  .status-dot.green {
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
  }
  .status-dot.orange {
    background: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
  }
  .status-dot.purple {
    background: #a855f7;
    box-shadow: 0 0 8px #a855f7;
  }
  .status-dot.red {
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
  }
  .visualizer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    height: 38px;
    width: 100%;
  }
  .bar {
    width: 5px;
    height: 6px;
    background: #60a5fa;
    border-radius: 3px;
    transition: height 0.08s ease, background 0.2s ease;
  }
  .transcript-box {
    width: 100%;
    max-height: 200px;
    min-height: 120px;
    overflow-y: auto;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-size: 14px;
  }
  .bubble {
    padding: 10px 14px;
    border-radius: 14px;
    max-width: 88%;
    line-height: 1.45;
    word-break: break-word;
  }
  .bubble.user {
    align-self: flex-end;
    background: #2563eb;
    color: #ffffff;
    border-bottom-right-radius: 2px;
  }
  .bubble.user.interim {
    background: rgba(37, 99, 235, 0.7);
    border: 1px dashed rgba(255, 255, 255, 0.4);
  }
  .bubble.assistant {
    align-self: flex-start;
    background: rgba(255, 255, 255, 0.12);
    color: #e2e8f0;
    border-bottom-left-radius: 2px;
  }
  .metrics-badge {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 5px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .controls-row {
    width: 100%;
    display: flex;
    gap: 10px;
  }
  .btn-call {
    flex: 1;
    padding: 14px 20px;
    border: none;
    border-radius: 14px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.2s ease;
    background: linear-gradient(135deg, #10b981, #059669);
    color: #ffffff;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35);
  }
  .btn-call:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
  }
  .btn-call.in-call {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.35);
  }
  .btn-interrupt {
    padding: 14px 18px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 14px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
    transition: all 0.2s ease;
  }
  .btn-interrupt:hover {
    background: rgba(239, 68, 68, 0.4);
    color: #ffffff;
  }
</style>
</head>
<body>

<div class="call-card">
  <div class="avatar-container" id="avatar">
    <div class="avatar-icon" id="avatar-icon">🎙️</div>
  </div>

  <div class="status-badge">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">Ready to Connect</span>
  </div>

  <div class="visualizer" id="visualizer">
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
    <div class="bar"></div>
  </div>

  <div class="transcript-box" id="transcript-box">
    <div style="color: #64748b; font-style: italic; text-align: center; padding: 10px;">
      Click "Start Voice Call" to stream your voice live with instant responses and barge-in.
    </div>
  </div>

  <div class="controls-row">
    <button class="btn-call" id="call-btn" onclick="toggleCall()">
      <span id="call-btn-text">📞 Start Voice Call</span>
    </button>
    <button class="btn-interrupt" id="interrupt-btn" onclick="triggerBargeIn()" style="display: none;">
      ⚡ Interrupt
    </button>
  </div>
</div>

<script>
  let inCall = false;
  let ws = null;
  let audioContext = null;
  let mediaStream = null;
  let audioInputNode = null;
  let scriptProcessor = null;
  let analyserNode = null;

  // Real-time audio streaming
  let isAssistantSpeaking = false;
  let activeTurnId = 0;
  let currentLiveUserBubble = null;
  let currentAssistantBubble = null;
  const audioQueue = [];
  let isPlayingAudio = false;
  let activeAudioSource = null;

  const avatar = document.getElementById('avatar');
  const avatarIcon = document.getElementById('avatar-icon');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  const callBtn = document.getElementById('call-btn');
  const callBtnText = document.getElementById('call-btn-text');
  const interruptBtn = document.getElementById('interrupt-btn');
  const transcriptBox = document.getElementById('transcript-box');
  const bars = document.querySelectorAll('.visualizer .bar');

  function setStatus(state, text) {
    statusText.innerText = text;
    statusDot.className = 'status-dot ' + (
      state === 'listening' ? 'green' :
      state === 'processing' ? 'orange' :
      state === 'speaking' ? 'purple' :
      state === 'interrupted' ? 'red' : ''
    );
    avatar.className = 'avatar-container ' + (
      state === 'listening' ? 'active' :
      state === 'speaking' ? 'speaking' :
      state === 'interrupted' ? 'interrupted' : ''
    );
    avatarIcon.innerText = state === 'speaking' ? '🗣️' : state === 'processing' ? '⚡' : state === 'interrupted' ? '✋' : '🎙️';
    interruptBtn.style.display = (inCall && isAssistantSpeaking) ? 'flex' : 'none';
  }

  function addMessage(role, text, metrics) {
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + role;
    bubble.innerText = text;
    if (metrics) {
      const mDiv = document.createElement('div');
      mDiv.className = 'metrics-badge';
      mDiv.innerText = `STT: ${metrics.stt_time.toFixed(2)}s | TTFT: ${metrics.ttft.toFixed(2)}s | TTFA: ${metrics.ttfa.toFixed(2)}s | Turn: ${metrics.total_time.toFixed(2)}s`;
      bubble.appendChild(mDiv);
    }
    transcriptBox.appendChild(bubble);
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
    return bubble;
  }

  function updateInterimUserBubble(text) {
    if (!currentLiveUserBubble) {
      currentLiveUserBubble = document.createElement('div');
      currentLiveUserBubble.className = 'bubble user interim';
      transcriptBox.appendChild(currentLiveUserBubble);
    }
    currentLiveUserBubble.innerText = text + ' ✍️';
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
  }

  function finalizeUserBubble(text) {
    if (currentLiveUserBubble) {
      currentLiveUserBubble.className = 'bubble user';
      currentLiveUserBubble.innerText = text;
      currentLiveUserBubble = null;
    } else {
      addMessage('user', text);
    }
  }

  function appendStreamingToken(token) {
    if (!currentAssistantBubble) {
      currentAssistantBubble = document.createElement('div');
      currentAssistantBubble.className = 'bubble assistant';
      transcriptBox.appendChild(currentAssistantBubble);
    }
    currentAssistantBubble.innerText += token;
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
  }

  function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  function floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output.buffer;
  }

  function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  async function toggleCall() {
    if (!inCall) {
      await startCall();
    } else {
      endCall();
    }
  }

  function triggerBargeIn() {
    if (!inCall) return;
    // 1. Immediately halt audio playback
    stopAssistantPlayback();
    // 2. Notify backend
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'interrupt' }));
    }
    setStatus('listening', '🎙️ Interrupted! Listening to you...');
  }

  function stopAssistantPlayback() {
    audioQueue.length = 0;
    isPlayingAudio = false;
    isAssistantSpeaking = false;
    if (activeAudioSource) {
      try { activeAudioSource.stop(); } catch(e) {}
      activeAudioSource = null;
    }
    interruptBtn.style.display = 'none';
  }

  async function startCall() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 16000
        }
      });
    } catch (err) {
      alert('Microphone permission error: ' + err.message);
      return;
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    const wsHost = window.location.hostname || '127.0.0.1';
    const wsUrl = `ws://${wsHost}:__WS_PORT__`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      inCall = true;
      callBtn.className = 'btn-call in-call';
      callBtnText.innerText = '🔴 End Call';
      transcriptBox.innerHTML = '';
      setStatus('listening', 'Connected! Start speaking...');
      setupStreamingAudioCapture();
    };

    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'interim_transcript') {
          updateInterimUserBubble(data.text);
          setStatus('listening', '🎙️ Listening...');
        } else if (data.type === 'user_transcript') {
          finalizeUserBubble(data.text);
          setStatus('processing', '⚡ Thinking...');
          currentAssistantBubble = null;
        } else if (data.type === 'assistant_start') {
          isAssistantSpeaking = true;
          activeTurnId = data.turn_id || activeTurnId;
          setStatus('speaking', '🗣️ Assistant speaking...');
        } else if (data.type === 'assistant_token') {
          appendStreamingToken(data.token);
        } else if (data.type === 'audio_chunk') {
          queueAudioChunk(data.audio, data.turn_id);
        } else if (data.type === 'interrupted') {
          stopAssistantPlayback();
          setStatus('interrupted', 'Interrupted');
          setTimeout(() => {
            if (inCall && !isAssistantSpeaking) setStatus('listening', '🎙️ Listening...');
          }, 300);
        } else if (data.type === 'turn_complete') {
          if (data.metrics && currentAssistantBubble) {
            const mDiv = document.createElement('div');
            mDiv.className = 'metrics-badge';
            mDiv.innerText = `STT: ${data.metrics.stt_time.toFixed(2)}s | TTFT: ${data.metrics.ttft.toFixed(2)}s | TTFA: ${data.metrics.ttfa.toFixed(2)}s | Total: ${data.metrics.total_time.toFixed(2)}s`;
            currentAssistantBubble.appendChild(mDiv);
          }
          currentAssistantBubble = null;
          if (!isPlayingAudio && audioQueue.length === 0) {
            isAssistantSpeaking = false;
            setStatus('listening', '🎙️ Listening to you...');
          }
        } else if (data.type === 'error') {
          addMessage('assistant', '⚠️ ' + (data.message || 'Pipeline error'));
          stopAssistantPlayback();
          setStatus('listening', '🎙️ Listening to you...');
        }
      } catch (err) {
        console.error('Error handling WebSocket frame:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setStatus('', 'Connection error');
    };

    ws.onclose = () => {
      if (inCall) endCall();
    };
  }

  function setupStreamingAudioCapture() {
    audioInputNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 256;

    // Stream ultra-low latency 64ms PCM chunks (1024 samples at 16kHz)
    scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);

    scriptProcessor.onaudioprocess = (e) => {
      if (!inCall) return;
      const inputData = e.inputBuffer.getChannelData(0);

      // Volume calculation for visualizer & client barge-in detection
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);

      // Client-side instant barge-in detection: if assistant is speaking and user speaks loudly
      if (isAssistantSpeaking && rms > 0.045) {
        triggerBargeIn();
      }

      // Send PCM audio chunk over WebSocket in real time!
      if (ws && ws.readyState === WebSocket.OPEN) {
        const pcmBuffer = floatTo16BitPCM(inputData);
        const b64Data = arrayBufferToBase64(pcmBuffer);
        ws.send(JSON.stringify({ type: 'audio_chunk', data: b64Data }));
      }
    };

    audioInputNode.connect(analyserNode);
    audioInputNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    startVisualizerLoop();
  }

  function queueAudioChunk(b64Audio, turnId) {
    audioQueue.push({ audio: b64Audio, turnId: turnId });
    if (!isPlayingAudio) {
      playNextAudioChunk();
    }
  }

  function playNextAudioChunk() {
    if (audioQueue.length === 0) {
      isPlayingAudio = false;
      if (!isAssistantSpeaking) {
        setStatus('listening', '🎙️ Listening to you...');
      }
      return;
    }

    isPlayingAudio = true;
    isAssistantSpeaking = true;
    setStatus('speaking', '🗣️ Assistant speaking...');

    const item = audioQueue.shift();
    try {
      const arrayBuffer = base64ToArrayBuffer(item.audio);
      audioContext.decodeAudioData(
        arrayBuffer,
        (decodedBuffer) => {
          activeAudioSource = audioContext.createBufferSource();
          activeAudioSource.buffer = decodedBuffer;
          activeAudioSource.connect(audioContext.destination);
          activeAudioSource.onended = () => {
            activeAudioSource = null;
            playNextAudioChunk();
          };
          activeAudioSource.start(0);
        },
        (err) => {
          console.error('Audio decode error:', err);
          playNextAudioChunk();
        }
      );
    } catch (e) {
      console.error('Playback error:', e);
      playNextAudioChunk();
    }
  }

  function startVisualizerLoop() {
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function updateVisualizer() {
      if (!inCall) return;
      analyserNode.getByteFrequencyData(dataArray);

      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const avg = sum / bufferLength / 255;

      bars.forEach((bar, idx) => {
        const h = Math.max(4, Math.min(34, avg * 130 * (0.8 + (idx % 3) * 0.4)));
        bar.style.height = `${h}px`;
        bar.style.background = isAssistantSpeaking ? '#c084fc' : '#34d399';
      });

      requestAnimationFrame(updateVisualizer);
    }

    requestAnimationFrame(updateVisualizer);
  }

  function endCall() {
    inCall = false;
    stopAssistantPlayback();
    currentLiveUserBubble = null;
    currentAssistantBubble = null;

    if (scriptProcessor) {
      try { scriptProcessor.disconnect(); } catch(e) {}
    }
    if (audioInputNode) {
      try { audioInputNode.disconnect(); } catch(e) {}
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
    }
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
    }
    if (ws) {
      ws.close();
    }

    callBtn.className = 'btn-call';
    callBtnText.innerText = '📞 Start Voice Call';
    setStatus('', 'Call Ended');
    bars.forEach(b => b.style.height = '4px');
  }
</script>
</body>
</html>
