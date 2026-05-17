document.addEventListener('DOMContentLoaded', function () {

  const S = {
    jobId: null,
    jobStarted: false, // true once job_started WS event received
    lang: 'en',
    uiLang: 'en',
    track: null,
    library: JSON.parse(localStorage.getItem('maz_lib') || '[]'),
    ws: null,
    wsR: 0,
    timer: null,
    aiResult: null,
    aiMode: null,
    voicing: false,   // true while voice pipeline phase is active
    wfRegion: null,   // {startPct, endPct} when a region is selected, else null
    reqDuration: -1,  // requested duration of the current/last generation (-1 = auto)
    serverElapsed: null, // last `elapsed` value reported by server (seconds)
    serverSyncMs: null,  // Date.now() when serverElapsed was last received
    notFoundStreak: 0,   // consecutive /job/{id} 404s — used to clear stuck state
    vocalGender: 'auto', // auto | female | male | mixed
  };

  const $ = id => document.getElementById(id);
  const $$ = s => document.querySelectorAll(s);

  // Tracks the in-flight AI request so closing the modal cancels it
  let _aiAbort = null;
  function _startAIRequest() {
    if (_aiAbort) _aiAbort.abort();
    _aiAbort = new AbortController();
    return _aiAbort;
  }
  const audio = $('audio');

  /* Theme */
  (function initTheme() {
    const saved = localStorage.getItem('maz_theme') || 'dark';
    if (saved === 'light') document.documentElement.setAttribute('data-theme', 'light');
  })();

  function toggleTheme() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    if (isLight) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('maz_theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      localStorage.setItem('maz_theme', 'light');
    }
  }

  /* Skin */
  (function initSkin() {
    const saved = localStorage.getItem('maz_skin') || 'default';
    applySkin(saved, false);
  })();

  function applySkin(skin, save = true) {
    const html = document.documentElement;

    if (skin === 'default') {
      html.removeAttribute('data-skin');
      // Restore saved light/dark preference
      const savedTheme = localStorage.getItem('maz_theme') || 'dark';
      if (savedTheme === 'light') html.setAttribute('data-theme', 'light');
      else html.removeAttribute('data-theme');
    } else {
      html.setAttribute('data-skin', skin);
      // Terminal skins are self-contained — clear data-theme
      html.removeAttribute('data-theme');
    }

    // Update picker active state
    $$('.skin-opt').forEach(btn =>
      btn.classList.toggle('active', btn.dataset.skin === skin)
    );

    // Show/hide the light-dark toggle (irrelevant when a terminal skin is active)
    const themeSection = $('btn-theme')?.closest('.sidebar-section');
    if (themeSection) themeSection.style.display = skin === 'default' ? '' : 'none';

    if (save) localStorage.setItem('maz_skin', skin);
  }

  document.addEventListener('click', e => {
    if (e.target.closest('#btn-theme')) toggleTheme();
    const skinBtn = e.target.closest('.skin-opt');
    if (skinBtn) applySkin(skinBtn.dataset.skin);
  });

  /* Nav */
  $$('.sidebar-item').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.sidebar-item').forEach(b => b.classList.remove('active'));
      $$('.page').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      $(`page-${btn.dataset.view}`).classList.add('active');
      if (btn.dataset.view === 'library') { renderLib(); loadStats(); }
    });
  });

  /* WebSocket */
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    S.ws = ws;
    ws.onopen = () => {
      S.wsR = 0;
      log('Connected', 'success');
      checkHealth();
      checkAI();   // report AI model status in the activity log
      // Re-sync job state if a job was in progress when WS dropped
      if (S.jobId) {
        fetch(`/job/${S.jobId}`)
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (!data || !S.jobId) return;
            if (data.status === 'completed' && data.audio_url) {
              handleWS({ type: 'job_completed', job_id: S.jobId, audio_url: data.audio_url, seed: -1 });
            } else if (data.status === 'failed') {
              handleWS({ type: 'job_failed', job_id: S.jobId, error: data.error || 'Generation failed' });
            } else if (data.status === 'generating') {
              S.jobStarted = true;
              startTimer();
            }
          })
          .catch(() => {});
      }
    };
    ws.onmessage = e => { try { handleWS(JSON.parse(e.data)); } catch (_) {} };
    ws.onclose = () => {
      S.wsR++;
      const d = Math.min(8000, S.wsR * 1200);
      log(`Reconnecting in ${(d / 1000).toFixed(0)}s`, 'warn');
      $('status-dot').className = 'status-dot warn';
      $('status-text').textContent = 'Reconnecting';
      setTimeout(connectWS, d);
    };
    ws.onerror = () => ws.close();
  }

  function handleWS(msg) {
    switch (msg.type) {
      case 'queue_update': updateQueue(msg); break;
      case 'training_progress': handleTrainingProgress(msg); break;
      case 'job_started':
        if (msg.job_id !== S.jobId) break;
        S.jobStarted = true;
        setLabel('Generating');
        log('Generation started', 'success');
        startTimer(); // safe: guarded against double-start inside startTimer()
        break;
      case 'job_progress':
        if (msg.job_id !== S.jobId) break;
        S.serverElapsed = msg.elapsed;
        S.serverSyncMs = Date.now();
        if (msg.status === 'applying voice') {
          S.voicing = true;
          setLabel(`Applying voice ${msg.elapsed}s`);
          break;
        }
        setLabel(`Generating ${msg.elapsed}s`);
        break;
      case 'job_completed':
        if (msg.job_id !== S.jobId) break;
        S.voicing = false;
        clearInterval(S.timer);
        S.track = {
          id: Date.now(),
          job_id: S.jobId,
          prompt: $('prompt').value.trim(),
          lyrics: $('lyrics').value.trim() || null,
          duration: parseInt($('duration').value),
          seed: msg.seed,
          audio_url: msg.audio_url,
          created: new Date().toISOString(),
        };
        loadTrack(S.track);
        resetBtn();
        S.jobId = null;
        log('Track ready', 'success');
        toast('Track ready — hit play!', 'success');
        maybeNotify('Track ready!', (S.track.prompt || '').slice(0, 80) || 'Your track is ready to play.');
        break;
      case 'job_cancelled':
        // Server confirms cancellation — UI is already reset by abortGeneration()
        break;
      case 'job_failed':
        if (msg.job_id !== S.jobId) break;
        S.voicing = false;
        clearInterval(S.timer);
        log(`Failed: ${msg.error}`, 'error');
        toast(`Generation failed: ${msg.error}`, 'error', 6000);
        maybeNotify('Generation failed', (msg.error || 'Track generation failed.').slice(0, 80));
        resetBtn();
        S.jobId = null;
        break;
      case 'voice_sections_progress':
        if (!S.track || msg.job_id !== S.track.job_id) break;
        $('vs-progress-msg') && ($('vs-progress-msg').textContent = msg.msg || 'Processing…');
        break;
      case 'voice_sections_done':
        if (!S.track || msg.job_id !== S.track.job_id) break;
        $('vs-progress')?.classList.add('hidden');
        $('btn-apply-sections') && ($('btn-apply-sections').disabled = false);
        S.track.audio_url = msg.audio_url;
        audio.src = msg.audio_url;
        audio.load();
        log('Voice sections applied — reloaded audio', 'success');
        toast('Voice sections applied', 'success');
        break;
      case 'voice_sections_failed':
        if (!S.track || msg.job_id !== S.track.job_id) break;
        if ($('btn-apply-sections')) $('btn-apply-sections').disabled = false;
        _vsPanelMsg(`Failed: ${msg.error || 'unknown error'}`, 5000);
        log(`Voice sections failed: ${msg.error}`, 'error');
        toast(`Voice sections failed: ${msg.error || 'unknown error'}`, 'error', 6000);
        break;
    }
  }

  function updateQueue(msg) {
    const ql = msg.queue_length;
    const aj = msg.active_job;
    const badge = $('q-badge');
    const body = $('q-body');
    if (aj?.job_id) {
      badge.textContent = t('badge_active');
      badge.classList.add('on');
      const pct = Math.min(90, ((aj.elapsed || 0) / 90) * 100).toFixed(1);
      body.innerHTML = `
        <div class="q-active">
          <div class="q-title">${escHtml(aj.prompt || '')}</div>
          <div class="q-meta">${aj.elapsed || 0}s · ${ql} waiting</div>
          <div class="q-bar"><div class="q-bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    } else {
      badge.textContent = 'idle';
      badge.classList.remove('on');
      body.innerHTML = `<span class="side-empty">${ql > 0 ? `${ql} waiting` : 'No active generation'}</span>`;
    }
    if (S.jobId && ql > 0 && !aj?.job_id) setLabel(`In queue #${ql}`);
  }

  /* Health */
  async function checkHealth() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      const ok = d.acestep_backend === 'ok';
      $('status-dot').className = 'status-dot ' + (ok ? 'online' : 'warn');
      $('status-text').textContent = ok ? 'Online' : 'Backend offline';
      if (!ok) log('AceStep not running on :8001', 'warn');
    } catch {
      $('status-dot').className = 'status-dot offline';
      $('status-text').textContent = 'Offline';
    }
  }
  setInterval(checkHealth, 30_000);

  /* AI status check */
  let _aiStatusCache = null;
  let _aiStatusTime  = 0;

  async function checkAI() {
    const now = Date.now();
    if (_aiStatusCache && now - _aiStatusTime < 60_000) return _aiStatusCache;
    try {
      const r = await fetch('/ai/status');
      _aiStatusCache = await r.json();
      _aiStatusTime  = now;
      if (_aiStatusCache.available) {
        log(`AI ready — model: ${_aiStatusCache.model}`, 'ai');
      } else {
        log(`AI offline — ${_aiStatusCache.reason}`, 'warn');
      }
    } catch {
      _aiStatusCache = { available: false, reason: 'Server not reachable' };
    }
    return _aiStatusCache;
  }


  /* Sliders */
  function bindSlider(id, valId, fmt) {
    const sl = $(id), vl = $(valId);
    if (!sl || !vl) return;
    const up = () => vl.textContent = fmt(sl.value);
    sl.addEventListener('input', up); up();
  }
  bindSlider('duration', 'dur-val', v => {
    const s = parseInt(v);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60), r = s % 60;
    return r ? `${m}m ${r}s` : `${m}m`;
  });
  bindSlider('guidance', 'guid-val', v => parseFloat(v).toFixed(1));
  bindSlider('steps', 'steps-val', v => v);

  /* Auto-duration from lyrics */
  function applyAutoDuration() {
    const chk = $('auto-duration');
    const slider = $('duration');
    const label = $('auto-dur-label');
    const val = $('dur-val');
    if (!chk || !slider) return;
    if (!chk.checked) {
      slider.disabled = false;
      label.classList.remove('active');
      val.textContent = fmtDur(parseInt(slider.value));
      return;
    }
    // Auto mode: ACE-Step LLM determines duration from lyrics via CoT reasoning
    slider.disabled = true;
    label.classList.add('active');
    val.textContent = 'AI';
  }

  $('auto-duration').addEventListener('change', applyAutoDuration);
  $('lyrics').addEventListener('input', () => {
    if ($('auto-duration').checked) applyAutoDuration();
  });

  /* Prompt char count */
  $('prompt').addEventListener('input', () => {
    $('char-count').textContent = $('prompt').value.length;
  });

  /* ── Prompt / Lyrics history dropdowns ──────────────────────── */
  {
    let histCache = null; // fetched once per page load, refreshed on open

    async function loadHistCache() {
      try {
        const r = await fetch('/history/prompts?limit=30');
        if (r.ok) histCache = await r.json();
      } catch (_) {}
    }
    loadHistCache(); // prefetch quietly in background

    function closeAllHistDropdowns() {
      $('prompt-hist-dropdown').classList.add('hidden');
      $('lyrics-hist-dropdown').classList.add('hidden');
    }

    // Close when clicking outside
    document.addEventListener('click', e => {
      if (!e.target.closest('#btn-prompt-hist') &&
          !e.target.closest('#prompt-hist-dropdown') &&
          !e.target.closest('#btn-lyrics-hist') &&
          !e.target.closest('#lyrics-hist-dropdown')) {
        closeAllHistDropdowns();
      }
    });

    function buildPromptDropdown(items) {
      const dd = $('prompt-hist-dropdown');
      if (!items.length) {
        dd.innerHTML = '<div class="hist-empty">No history yet</div>';
        return;
      }
      dd.innerHTML = items.map((item, i) => `
        <div class="hist-item" data-idx="${i}">
          <div class="hist-item-prompt">${escHtml(item.prompt.slice(0, 80))}${item.prompt.length > 80 ? '…' : ''}</div>
          ${item.lyrics ? `<div class="hist-item-sub">+ lyrics</div>` : ''}
        </div>`).join('');
      dd.querySelectorAll('.hist-item').forEach(el => {
        el.addEventListener('click', () => {
          const item = items[parseInt(el.dataset.idx)];
          $('prompt').value = item.prompt;
          $('char-count').textContent = item.prompt.length;
          closeAllHistDropdowns();
          $('prompt').focus();
        });
      });
    }

    function buildLyricsDropdown(items) {
      const withLyrics = items.filter(i => i.lyrics && i.lyrics.trim());
      const dd = $('lyrics-hist-dropdown');
      if (!withLyrics.length) {
        dd.innerHTML = '<div class="hist-empty">No lyrics history yet</div>';
        return;
      }
      dd.innerHTML = withLyrics.map((item, i) => `
        <div class="hist-item" data-idx="${i}">
          <div class="hist-item-prompt">${escHtml(item.prompt.slice(0, 60))}${item.prompt.length > 60 ? '…' : ''}</div>
          <div class="hist-item-sub">${escHtml(item.lyrics.slice(0, 60))}${item.lyrics.length > 60 ? '…' : ''}</div>
        </div>`).join('');
      dd.querySelectorAll('.hist-item').forEach(el => {
        el.addEventListener('click', () => {
          const item = withLyrics[parseInt(el.dataset.idx)];
          $('lyrics').value = item.lyrics;
          closeAllHistDropdowns();
          $('lyrics').focus();
        });
      });
    }

    $('btn-prompt-hist').addEventListener('click', async e => {
      e.stopPropagation();
      const dd = $('prompt-hist-dropdown');
      if (!dd.classList.contains('hidden')) { dd.classList.add('hidden'); return; }
      $('lyrics-hist-dropdown').classList.add('hidden');
      if (!histCache) await loadHistCache();
      buildPromptDropdown(histCache?.prompts || []);
      dd.classList.remove('hidden');
    });

    $('btn-lyrics-hist').addEventListener('click', async e => {
      e.stopPropagation();
      const dd = $('lyrics-hist-dropdown');
      if (!dd.classList.contains('hidden')) { dd.classList.add('hidden'); return; }
      $('prompt-hist-dropdown').classList.add('hidden');
      if (!histCache) await loadHistCache();
      buildLyricsDropdown(histCache?.prompts || []);
      dd.classList.remove('hidden');
    });
  }

  /* Seed */
  $('btn-rand').addEventListener('click', () => {
    $('seed').value = Math.floor(Math.random() * 2147483647);
  });

  /* Tags */
  $$('.tag').forEach(t => t.addEventListener('click', () => t.classList.toggle('on')));
  function getTags() { return [...$$('.tag.on')].map(t => t.dataset.style).join(', '); }

  /* Language selector */
  $$('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.lang-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      S.lang = btn.dataset.lang;
      const customInput = $('custom-lang-input');
      if (btn.dataset.lang === 'custom') {
        customInput.classList.remove('hidden');
        customInput.focus();
      } else {
        customInput.classList.add('hidden');
        customInput.value = '';
      }
      const names = { en: 'English', kk: 'Kazakh', ru: 'Russian', tr: 'Turkish', custom: 'Custom' };
      log(`Lyrics language set to ${names[S.lang] || S.lang}`, 'ai');
    });
  });

  /* Vocal gender selector */
  $$('.vocal-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.vocal-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      S.vocalGender = btn.dataset.vocal;
      log(`${t('label_vocals')}: ${t('vocal_' + S.vocalGender)}`, 'ai');
    });
  });

  /* AI: Enhance */
  $('btn-enhance').addEventListener('click', async () => {
    const p = $('prompt').value.trim();
    if (!p) { log('Enter a prompt first', 'warn'); return; }
    S.aiMode = 'enhance';
    $('modal-title').textContent = 'Enhance Prompt';
    showThinking('Enhancing your prompt with AI...');
    const _enhanceCtrl = _startAIRequest();
    try {
      const r = await callAI(`You are an expert music prompt engineer. Enhance the original prompt into a precise, consistent, and concise description for an AI music generator.
Focus on exact musical attributes such as genre, mood, tempo, instruments, and production style.
Do NOT generate long paragraphs or random topics. Return a single cohesive sentence or comma-separated list.
Return ONLY the enhanced prompt text.

Original: "${p}"`, '', _enhanceCtrl.signal, 'enhance');
      if (_enhanceCtrl.signal.aborted) return;
      if (!r) throw new Error('AI returned an empty response. Try again.');
      S.aiResult = r;
      showResult(r);
      log('Prompt enhanced', 'ai');
    } catch (e) {
      if (e.name === 'AbortError') return;
      showAIError(e.message); log(`AI: ${e.message}`, 'error');
    }
  });

  /* AI: Lyrics */
  $('btn-write-lyrics').addEventListener('click', () => {
    $('lyrics-gen-row').classList.toggle('hidden');
    if (!$('lyrics-gen-row').classList.contains('hidden')) $('lyrics-theme').focus();
  });

  $('btn-gen-lyrics').addEventListener('click', async () => {
    const theme = $('lyrics-theme').value.trim() || $('prompt').value.trim();
    if (!theme) { log('Enter a theme or prompt first', 'warn'); return; }

    const activeLangBtn = document.querySelector('.lang-btn.active');
    const langCode = (activeLangBtn ? activeLangBtn.dataset.lang : null) || S.lang || 'en';

    // Validate custom language input before showing modal
    const customLangName = langCode === 'custom' ? $('custom-lang-input').value.trim() : '';
    if (langCode === 'custom' && !customLangName) {
      log('Enter a language name in the custom field first', 'warn');
      $('custom-lang-input').focus();
      return;
    }

    S.aiMode = 'lyrics';
    showThinking('Writing your lyrics...');
    const _lyricsCtrl = _startAIRequest();

    const uiToLyricLang = { en: 'english', kk: 'kazakh', ru: 'russian', tr: 'turkish', custom: 'custom' };
    const lang = uiToLyricLang[langCode] || 'english';

    const prompts = {
      english: `Write complete song lyrics ONLY in English about: "${theme}"

Rules:
- Write ONLY in English — no other language, no translations, no explanations
- Use a consistent rhyme scheme (AABB or ABAB) with natural rhythm
- Each verse: 6-8 lines. Chorus: 4-6 lines. Bridge: 4 lines.
- Make it emotional, vivid, and suitable for a 3-4 minute song

Structure:

[Verse 1]

[Pre-Chorus]

[Chorus]

[Verse 2]

[Pre-Chorus]

[Chorus]

[Bridge]

[Chorus]`,

      kazakh: `Мына тақырыпта қазақ тілінде толық ән мәтінін жаз: "${theme}"

Міндетті ережелер:
- Әрбір жол тек қазақ тілінде болуы керек. Орысша немесе ағылшынша сөздер болмасын.
- Нақты қазақ сөздерін қолдан: мысалы "жүрек", "күн", "жер", "арман", "махаббат", "жол", "от", "су", "аспан", "жел".
- Ұйқас схемасы AABB (1-жол 2-жолмен, 3-жол 4-жолмен ұйқасады) немесе ABAB болуы керек.
- Жолдар табиғи ырғақпен оқылуы керек — 7-8 буынды сақта.
- Қазақ тіліне тән жалғауларды қолдан: -дай/-дей, -лы/-лі, -ды/-ді, -ған/-ген, -мен/-бен, -да/-де.
- Ешқандай аударма, комментарий немесе түсіндірме жазба.
- Толық аяқталған мәтін жаз — ешқашан "(мәтін осында)" сияқты нұсқаулар қалдырма.

Ән құрылымы:

[1-шумақ]
(6 жол, AABB ұйқасы)

[Қайырма]
(4 жол, AABB ұйқасы)

[2-шумақ]
(6 жол, AABB ұйқасы)

[Қайырма]
(4 жол, AABB ұйқасы)

[Кепіл]
(4 жол)

[Қайырма]
(4 жол, AABB ұйқасы)`,

      russian: `Напиши полный текст песни на русском языке на тему: "${theme}"

Правила:
- Пиши ТОЛЬКО на русском языке, без переводов и комментариев
- Используй чёткую рифмовку (ААББ или АБАБ) и естественный ритм
- Каждый куплет: 6-8 строк. Припев: 4-6 строк. Бридж: 4 строки.
- Текст должен быть живым, образным, подходящим для песни 3-4 минуты

Структура:

[Куплет 1]

[Припев]

[Куплет 2]

[Припев]

[Бридж]

[Припев]`,

      turkish: `Şu tema hakkında Türkçe tam bir şarkı sözü yaz: "${theme}"

Kurallar:
- YALNIZCA Türkçe yaz — çeviri, açıklama veya yorum ekleme
- Tutarlı bir kafiye düzeni kullan (AABB veya ABAB) ve doğal bir ritim kur
- Her kıta: 6-8 satır. Nakarat: 4-6 satır. Köprü: 4 satır.
- Duygusal, canlı ve 3-4 dakikalık bir şarkıya uygun sözler yaz

Yapı:

[Kıta 1]

[Ön Nakarat]

[Nakarat]

[Kıta 2]

[Ön Nakarat]

[Nakarat]

[Köprü]

[Nakarat]`,

      custom: `Write complete song lyrics ONLY in ${customLangName} about: "${theme}"

Rules:
- Write ONLY in ${customLangName} — no English, no other language, no translations, no explanations
- Use authentic vocabulary, idioms, and rhyme style native to ${customLangName}
- Use a consistent rhyme scheme (AABB or ABAB) with natural rhythm
- Each verse: 6-8 lines. Chorus: 4-6 lines. Bridge: 4 lines.
- Make it emotional, vivid, and suitable for a 3-4 minute song

Structure:

[Verse 1]

[Pre-Chorus]

[Chorus]

[Verse 2]

[Pre-Chorus]

[Chorus]

[Bridge]

[Chorus]`
    };

    const prompt = prompts[lang] || prompts.english;
    const langNames = { english: 'English', kazakh: 'Қазақша', russian: 'Русский', turkish: 'Türkçe' };
    const displayName = lang === 'custom' ? customLangName : (langNames[lang] || lang);
    $('modal-title').textContent = `Write Lyrics — ${displayName}`;

    // For custom language, send the actual name to the server so it picks the right system prompt.
    // For known languages, send the ISO code.
    const serverLang = lang === 'custom' ? customLangName : langCode;

    try {
      const r = await callAI(prompt, serverLang, _lyricsCtrl.signal);
      if (_lyricsCtrl.signal.aborted) return;
      if (!r) throw new Error('AI returned an empty response. Try again.');
      S.aiResult = r;
      showResult(r);
      log(`Lyrics written in ${displayName}`, 'ai');
    } catch (e) {
      if (e.name === 'AbortError') return;
      showAIError(e.message); log(`AI: ${e.message}`, 'error');
    }
  });


  async function callAI(prompt, language = '', signal = null, mode = 'lyrics') {
    const res = await fetch('/ai/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, language, mode }),
      signal,
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || res.statusText);
    }
    return (await res.json()).result;
  }

  function showThinking(msg) {
    $('ai-modal').classList.remove('hidden');
    $('modal-foot').classList.add('hidden');
    $('modal-body').innerHTML = `
      <div class="thinking">
        <div class="dots"><span></span><span></span><span></span></div>
        <p id="think-label">${msg}</p>
      </div>`;
  }

  function showAIError(msg) {
    // Keep modal open, show the error with guidance so the user knows what to do
    const isNoModel = /no local ai model/i.test(msg) || /no model/i.test(msg);
    const hint = isNoModel
      ? `<p style="opacity:.45;font-size:12px;line-height:1.5">
          To enable AI: install <strong>Ollama</strong> from <code>ollama.ai</code>,<br>
          then run: <code>ollama pull qwen3:8b</code>
        </p>`
      : `<p style="opacity:.45;font-size:12px;line-height:1.5">
          If this keeps happening, try restarting Ollama or the server.
        </p>`;
    $('modal-body').innerHTML = `
      <div style="text-align:center;padding:16px 8px">
        <svg viewBox="0 0 24 24" fill="none" style="width:36px;height:36px;opacity:.5;margin-bottom:10px">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.8"/>
          <path d="M12 8v4M12 16h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <p style="font-weight:600;margin-bottom:8px">AI unavailable</p>
        <p style="opacity:.65;font-size:13px;line-height:1.6;margin-bottom:12px">${escHtml(msg)}</p>
        ${hint}
      </div>`;
    $('modal-foot').classList.add('hidden');
  }
  function showResult(text) {
    $('modal-body').innerHTML = `<div class="ai-result">${escHtml(text)}</div>`;
    $('modal-foot').classList.remove('hidden');
  }
  function closeModal() {
    if (_aiAbort) { _aiAbort.abort(); _aiAbort = null; }
    $('ai-modal').classList.add('hidden');
    S.aiResult = null; S.aiMode = null;
  }
  $('modal-close').addEventListener('click', closeModal);
  $('modal-discard').addEventListener('click', closeModal);
  $('modal-apply').addEventListener('click', () => {
    if (!S.aiResult) return;
    if (S.aiMode === 'enhance') {
      $('prompt').value = S.aiResult;
      log('Prompt applied', 'success');
    } else {
      $('lyrics').value = S.aiResult;
      // Trigger input event so auto-duration recalculates from new lyrics
      $('lyrics').dispatchEvent(new Event('input'));
      $('lyrics-gen-row').classList.add('hidden');
      log('Lyrics applied', 'success');
    }
    closeModal();
  });

  /* Generate */
  $('btn-generate').addEventListener('click', async () => {
    const p = $('prompt').value.trim();
    if (!p) { log('Enter a prompt first', 'warn'); return; }
    if (S.jobId) return;

    const tags = getTags();
    const voiceOn = $('voice-toggle') && $('voice-toggle').checked;
    const voiceId = voiceOn && $('voice-profile-select') ? $('voice-profile-select').value : null;
    const reqDuration = $('auto-duration').checked ? -1 : parseInt($('duration').value);
    S.reqDuration = reqDuration;
    const payload = {
      prompt: tags ? `${p}, ${tags}` : p,
      lyrics: $('lyrics').value.trim() || null,
      duration: reqDuration,
      guidance_scale: parseFloat($('guidance').value),
      infer_steps: parseInt($('steps').value),
      seed: $('seed').value ? parseInt($('seed').value) : -1,
      bpm: $('bpm').value ? parseInt($('bpm').value) : null,
      key: $('key').value.trim() || null,
      language: S.lang,
      voice_id: voiceId || null,
      vocal_gender: S.vocalGender,
    };

    requestNotifPermission();
    $('btn-generate').disabled = true;
    $('gen-idle').classList.add('hidden');
    $('gen-busy').classList.remove('hidden');
    $('btn-cancel-gen').classList.remove('hidden');
    setLabel('Submitting');
    log(`Submitting: "${p.slice(0, 60)}${p.length > 60 ? '...' : ''}"`);

    try {
      const res = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.status === 429) { const e = await res.json(); throw new Error(e.detail); }
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || res.statusText); }
      const data = await res.json();
      S.jobId = data.job_id;
      S.jobStarted = false;
      setLabel(data.position === 1 ? 'Starting...' : `In queue #${data.position}`);
      log(`Queued at #${data.position} · est. ${data.estimated_wait_minutes} min`);
      startTimer(); // start fallback polling immediately, regardless of WS state
    } catch (err) {
      log(`Error: ${err.message}`, 'error');
      resetBtn();
    }
  });

  $('btn-cancel-gen').addEventListener('click', () => {
    log('Generation cancelled.', 'warn');
    toast('Generation cancelled', 'warn', 2500);
    abortGeneration();
  });

  function resetBtn() {
    $('btn-generate').disabled = false;
    $('gen-idle').classList.remove('hidden');
    $('gen-busy').classList.add('hidden');
    $('btn-cancel-gen').classList.add('hidden');
    clearInterval(S.timer);
    S.timer = null;
    S.jobStarted = false;
    S.serverElapsed = null;
    S.serverSyncMs = null;
    S.notFoundStreak = 0;
  }

  function abortGeneration() {
    const jobId = S.jobId;
    S.jobId   = null;
    S.voicing = false;
    resetBtn();
    if (jobId) fetch(`/job/${jobId}`, { method: 'DELETE' }).catch(() => {});
  }

  function setLabel(t) { $('gen-label').textContent = t; }
  function startTimer() {
    if (S.timer) return; // already running — prevent duplicate intervals
    let t = 0;
    S.timer = setInterval(async () => {
      t++;
      // Display: prefer server-reported elapsed (interpolated for smoothness)
      // so the counter never jumps backward when WS messages arrive.
      if (S.jobStarted && !S.voicing) {
        let displaySecs;
        if (S.serverElapsed != null && S.serverSyncMs != null) {
          displaySecs = S.serverElapsed + Math.floor((Date.now() - S.serverSyncMs) / 1000);
        } else {
          displaySecs = t;
        }
        setLabel(`Generating ${displaySecs}s`);
      }

      // Every 15s poll the HTTP status endpoint as a fallback in case a WS
      // broadcast was lost (e.g. brief disconnect during completion/failure).
      if (t % 15 === 0 && S.jobId) {
        try {
          const res = await fetch(`/job/${S.jobId}`);
          if (res.ok) {
            S.notFoundStreak = 0;
            const data = await res.json();
            if (data.status === 'completed' && data.audio_url) {
              clearInterval(S.timer); S.timer = null;
              handleWS({ type: 'job_completed', job_id: S.jobId, audio_url: data.audio_url, seed: -1 });
            } else if (data.status === 'failed') {
              clearInterval(S.timer); S.timer = null;
              handleWS({ type: 'job_failed', job_id: S.jobId, error: data.error || 'Generation failed' });
            } else if (data.status === 'generating' && !S.jobStarted) {
              S.jobStarted = true; // job moved from queue to active between polls
            }
          } else if (res.status === 404) {
            // Job is gone from the server (likely server restart or rotated out
            // of recently_finished). Two consecutive 404s → recover the UI so
            // the user isn't blocked from starting a new generation.
            S.notFoundStreak++;
            if (S.notFoundStreak >= 2) {
              clearInterval(S.timer); S.timer = null;
              log('Lost contact with the previous job — resetting.', 'warn');
              handleWS({ type: 'job_failed', job_id: S.jobId, error: 'Job no longer tracked by server' });
            }
          }
        } catch (_) { /* network error — keep waiting */ }
      }

      // Two-phase timeout: scales with requested duration (matches server-side logic).
      //   Music generation phase  → server timeout + 60 s buffer
      //   Voice pipeline phase    → 1300s total (voice pipeline server timeout is 600s)
      const dur = (S.reqDuration > 0) ? S.reqDuration : 0;
      const serverLimit = Math.min(Math.max(600, dur * 15), 3600);
      const hardLimit = S.voicing ? 1300 : serverLimit + 60;
      if (t >= hardLimit && S.jobId) {
        const msg = S.voicing
          ? 'Voice conversion timed out. Your generated track (without voice) may still be in the library.'
          : 'Generation timed out — ACE-Step may be overloaded or unresponsive. Try again.';
        log(msg, 'error');
        abortGeneration();
      }
    }, 1000);
  }

  /* Load track */
  function loadTrack(track) {
    audio.src = track.audio_url;
    audio.load();

    const dur = fmtDur(track.duration);
    const displayTitle = (track.title || track.prompt || '');

    $('np-empty').classList.add('hidden');
    $('np-loaded').classList.remove('hidden');
    $('np-badge').textContent = 'NEW';
    $('np-title').textContent = displayTitle.slice(0, 60) + (displayTitle.length > 60 ? '...' : '');
    $('np-meta').textContent = `${dur} · seed ${track.seed ?? '—'}`;
    drawArt($('np-art'), track.seed || 42, 260);

    $('player-bar').classList.remove('hidden');
    $('bar-title').textContent = displayTitle.slice(0, 50) + (displayTitle.length > 50 ? '...' : '');
    $('bar-meta').textContent = dur;
    drawArt($('bar-art'), track.seed || 42, 56);

    audio.addEventListener('loadedmetadata', () => {
      const d = fmt(audio.duration);
      $('t-total').textContent = d;
      $('bar-total').textContent = d;
      // If input duration was auto (-1) or differs from reality, update the displayed duration
      if (!track.duration || track.duration <= 0 || Math.abs(audio.duration - track.duration) > 2) {
        const realDur = fmtDur(Math.round(audio.duration));
        $('np-meta').textContent = `${realDur} · seed ${track.seed ?? '—'}`;
        $('bar-meta').textContent = realDur;
        track.duration = Math.round(audio.duration); // keep track in sync
      }
      drawWF();
    }, { once: true });

    audio.addEventListener('error', () => {
      setPlaying(false);
      log('Audio file not found — it may have been cleaned up by the server', 'error');
      toast('Could not load audio. Try regenerating the track.', 'error');
    }, { once: true });

    setPlaying(false);

    // Sync lyrics textarea to the loaded track so refreshSectionsPanel reads correctly
    if (track.lyrics != null) {
      $('lyrics').value = track.lyrics;
    }

    // Run audio analysis automatically
    runAnalysis(track);

    // Show voice sections panel (always — falls back to Whole Track if no tags)
    refreshSectionsPanel();
  }

  /* ── Voice Sections Panel ──────────────────────────────────── */
  let _voiceProfilesCache = [];   // kept in sync by loadVoiceProfiles()
  let _secPreviewAudio = null, _secPreviewTimer = null;
  function _profileColor(name) {
    if (!name) return null;
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
    return `hsl(${h % 360},60%,55%)`;
  }

  function parseSections(lyrics) {
    if (!lyrics) return [];
    const tags = [];
    for (const line of lyrics.split('\n')) {
      const m = line.match(/^\s*\[([^\]]+)\]\s*$/);
      if (m) tags.push(m[1].trim());
    }
    return tags;
  }

  function buildSectionRows(sections) {
    const list = $('vs-sections-list');
    if (!list) return;
    const svcProfiles = _voiceProfilesCache.filter(p => p.type !== 'rvc');
    const opts = `<option value="">— keep original —</option>` +
      svcProfiles.map(p => `<option value="${escHtml(p.voice_id)}">${escHtml(p.name)}</option>`).join('');
    const dur = S.track?.duration || 0;
    const n   = sections.length || 1;
    list.innerHTML = sections.map((label, i) => {
      const start = ((i / n) * dur).toFixed(2);
      return `
      <div class="vs-section-row" data-section-idx="${i}" style="border-left:3px solid transparent">
        <span class="vs-section-label">${escHtml(label)}</span>
        <button class="vs-preview-btn" data-start="${start}" title="Preview 3s of this section">▶</button>
        <select class="vs-section-select" data-section-idx="${i}">${opts}</select>
      </div>`;
    }).join('');

    // Color-code row border when a voice profile is selected
    list.querySelectorAll('.vs-section-select').forEach(sel => {
      const syncColor = () => {
        const row  = sel.closest('.vs-section-row');
        const name = sel.options[sel.selectedIndex]?.textContent || '';
        row.style.borderLeftColor = sel.value ? (_profileColor(name) || 'transparent') : 'transparent';
      };
      sel.addEventListener('change', syncColor);
    });

    // Section audio preview (3s clip from estimated start)
    list.querySelectorAll('.vs-preview-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!S.track) return;
        const hasSec = !!S.track.voice_sections_path;
        const url    = hasSec ? `/audio/sections_${S.track.job_id}` : `/audio/${S.track.job_id}`;
        const start  = parseFloat(btn.dataset.start) || 0;
        if (_secPreviewAudio) { _secPreviewAudio.pause(); clearTimeout(_secPreviewTimer); }
        // Reset any other active preview buttons
        list.querySelectorAll('.vs-preview-btn').forEach(b => { b.textContent = '▶'; b.classList.remove('playing'); });
        _secPreviewAudio = new Audio(url);
        _secPreviewAudio.currentTime = start;
        _secPreviewAudio.volume = 0.8;
        _secPreviewAudio.play().catch(() => {});
        btn.textContent = '■';
        btn.classList.add('playing');
        const reset = () => { btn.textContent = '▶'; btn.classList.remove('playing'); };
        _secPreviewTimer = setTimeout(() => { _secPreviewAudio?.pause(); reset(); }, 3000);
        _secPreviewAudio.addEventListener('pause', reset, { once: true });
        _secPreviewAudio.addEventListener('ended', reset, { once: true });
      });
    });
  }

  function refreshSectionsPanel() {
    const panel = $('voice-sections-panel');
    if (!panel || !S.track) return;
    const lyrics = S.track.lyrics || $('lyrics').value.trim();
    const sections = parseSections(lyrics);
    // Always show the panel — fall back to a single "Whole Track" section
    // so voice conversion is available even for tracks without structured lyrics.
    panel.classList.remove('hidden');
    buildSectionRows(sections.length ? sections : ['Whole Track']);
  }

  // Toggle open/close — preserve user's selections when reopening
  $('vs-header')?.addEventListener('click', () => {
    const body = $('vs-body');
    const header = $('vs-header');
    if (!body) return;
    const isOpen = !body.classList.contains('hidden');
    body.classList.toggle('hidden', isOpen);
    header.classList.toggle('open', !isOpen);
    if (isOpen) return; // closing — keep rows intact
    if (!S.track) return;
    const lyrics = S.track.lyrics || $('lyrics').value.trim();
    const labels = parseSections(lyrics);
    // Preserve user's current selections before rebuilding rows
    const saved = {};
    document.querySelectorAll('.vs-section-select').forEach(sel => {
      if (sel.value) saved[sel.dataset.sectionIdx] = sel.value;
    });
    buildSectionRows(labels);
    // Restore selections that are still valid in the new option set
    document.querySelectorAll('.vs-section-select').forEach(sel => {
      const v = saved[sel.dataset.sectionIdx];
      if (v) sel.value = v; // browser ignores unknown values — safe
    });
  });

  function _vsPanelMsg(text, autoHide) {
    const prog = $('vs-progress');
    const msg  = $('vs-progress-msg');
    if (!prog || !msg) return;
    msg.textContent = text;
    prog.classList.remove('hidden');
    if (autoHide) setTimeout(() => prog.classList.add('hidden'), autoHide);
  }

  // Apply button
  $('btn-apply-sections')?.addEventListener('click', async () => {
    if (!S.track?.job_id) {
      _vsPanelMsg('No track loaded.', 3000);
      return;
    }
    const lyrics = S.track.lyrics || $('lyrics').value.trim();
    const sectionLabels = parseSections(lyrics);

    const sections = sectionLabels.map((label, i) => {
      const sel = document.querySelector(`.vs-section-select[data-section-idx="${i}"]`);
      return { label, voice_id: sel?.value || null };
    });

    const hasVoice = sections.some(s => s.voice_id);
    if (!hasVoice) {
      _vsPanelMsg('Select a voice profile for at least one section.', 3500);
      return;
    }

    const btn = $('btn-apply-sections');
    btn.disabled = true;
    _vsPanelMsg('Submitting…');
    log('Applying voice sections…');

    try {
      const res = await fetch('/voice/sections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: S.track.job_id, sections }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || res.statusText);
      }
      _vsPanelMsg('Processing… this may take a few minutes.');
    } catch (err) {
      log(`Voice sections failed: ${err.message}`, 'error');
      _vsPanelMsg(`Error: ${err.message}`, 4000);
      btn.disabled = false;
    }
  });

  function fmtDur(s) {
    if (!s || s <= 0) return 'auto';
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60), r = s % 60;
    return r ? `${m}m ${r}s` : `${m}m`;
  }

  /* ── Audio Analysis ──────────────────────────────────────── */
  async function runAnalysis(track) {
    const card = $('analysis-card');
    const loading = $('analysis-loading');
    const data = $('analysis-data');
    if (!card) return;

    card.classList.remove('hidden');
    loading.classList.remove('hidden');
    data.classList.add('hidden');

    const taskId = track.audio_url.split('/').pop();

    try {
      const res = await fetch(`/analyze/${taskId}`);
      if (!res.ok) throw new Error(res.statusText);
      const a = await res.json();

      // Metrics
      $('a-bpm').textContent = Math.round(a.bpm);
      $('a-key').textContent = a.key;
      $('a-mood').textContent = a.mood;
      $('a-energy').textContent = Math.round(a.energy * 100) + '%';

      // Spectrum bars
      const bass = a.spectrum.bass;
      const mid = a.spectrum.mid;
      const treble = a.spectrum.treble;

      $('spec-bass').style.height = bass + '%';
      $('spec-mid').style.height = mid + '%';
      $('spec-treble').style.height = treble + '%';
      $('spec-bass-val').textContent = bass + '%';
      $('spec-mid-val').textContent = mid + '%';
      $('spec-treble-val').textContent = treble + '%';

      loading.classList.add('hidden');
      data.classList.remove('hidden');

      track.analysis = a;
      log(`Analysis: ${Math.round(a.bpm)} BPM · ${a.key} · ${a.mood}`, 'success');

    } catch (e) {
      card.classList.add('hidden');
      log(`Analysis failed: ${e.message}`, 'error');
    }
  }

  /* Playback */
  function play() { audio.play(); setPlaying(true); }
  function pause() { audio.pause(); setPlaying(false); }

  $('btn-play').addEventListener('click', () => audio.paused ? play() : pause());
  $('btn-back').addEventListener('click', () => { audio.currentTime = Math.max(0, audio.currentTime - 10); });
  $('btn-fwd').addEventListener('click', () => { audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 10); });
  $('bar-play').addEventListener('click', () => audio.paused ? play() : pause());
  $('bar-back').addEventListener('click', () => { audio.currentTime = Math.max(0, audio.currentTime - 10); });
  $('bar-fwd').addEventListener('click', () => { audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 10); });

  audio.addEventListener('timeupdate', () => {
    const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
    $('wf-fill').style.width = pct + '%';
    $('wf-head').style.left = pct + '%';
    $('bar-fill').style.width = pct + '%';
    $('t-now').textContent = fmt(audio.currentTime);
    $('bar-now').textContent = fmt(audio.currentTime);
  });
  audio.addEventListener('ended', () => setPlaying(false));

  // ── Waveform — seek, drag-scrub, hover ghost + tooltip, shift+drag region ──
  (function initWaveform() {
    const wf          = $('np-waveform');
    const ghost       = $('wf-ghost');
    const tooltip     = $('wf-tooltip');
    const regionEl    = $('wf-region');
    const regionLabel = $('wf-region-label');
    let dragging      = false;
    let selecting     = false;  // true during shift+drag region selection
    let regionAnchor  = 0;      // pct where shift+drag started

    function pctFromEvent(e) {
      const r = wf.getBoundingClientRect();
      return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    }
    function seekTo(pct) {
      if (audio.duration) audio.currentTime = pct * audio.duration;
    }
    function updateGhost(e) {
      const pct = pctFromEvent(e);
      const px  = pct * wf.getBoundingClientRect().width;
      ghost.style.left    = pct * 100 + '%';
      tooltip.style.left  = px + 'px';
      tooltip.textContent = audio.duration ? fmt(pct * audio.duration) : '';
    }
    function renderRegion() {
      const r = S.wfRegion;
      if (!r) { regionEl.classList.remove('visible'); return; }
      const left  = Math.min(r.startPct, r.endPct) * 100;
      const width = Math.abs(r.endPct - r.startPct) * 100;
      regionEl.style.left  = left + '%';
      regionEl.style.width = width + '%';
      regionEl.classList.add('visible');
      if (audio.duration) {
        const t0 = fmt(Math.min(r.startPct, r.endPct) * audio.duration);
        const t1 = fmt(Math.max(r.startPct, r.endPct) * audio.duration);
        regionLabel.textContent  = `${t0} — ${t1}`;
        regionLabel.style.left   = left + '%';
      }
    }
    function clearRegion() {
      S.wfRegion = null;
      regionEl.classList.remove('visible');
      regionLabel.textContent = '';
    }

    wf.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      if (e.shiftKey) {
        // Start region selection
        selecting    = true;
        regionAnchor = pctFromEvent(e);
        S.wfRegion   = { startPct: regionAnchor, endPct: regionAnchor };
        renderRegion();
        wf.style.cursor = 'crosshair';
      } else {
        // Plain click — clear region and seek
        clearRegion();
        dragging = true;
        seekTo(pctFromEvent(e));
        wf.style.cursor = 'grabbing';
      }
    });
    window.addEventListener('mousemove', e => {
      if (dragging) { seekTo(pctFromEvent(e)); return; }
      if (selecting) {
        S.wfRegion = { startPct: regionAnchor, endPct: pctFromEvent(e) };
        renderRegion();
      }
    });
    window.addEventListener('mouseup', () => {
      if (dragging)  { dragging  = false; wf.style.cursor = ''; }
      if (selecting) {
        selecting = false;
        wf.style.cursor = '';
        // Discard trivially-small selections (< 0.5% of total)
        const r = S.wfRegion;
        if (r && Math.abs(r.endPct - r.startPct) < 0.005) clearRegion();
      }
    });
    wf.addEventListener('mousemove', updateGhost);
    wf.addEventListener('mouseleave', () => {
      ghost.style.left   = '-9999px';
      tooltip.style.left = '-9999px';
    });
    // Double-click clears any active region
    wf.addEventListener('dblclick', clearRegion);
  })();
  $('bar-track').addEventListener('click', e => {
    if (!audio.duration) return;
    const r = $('bar-track').getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
  });

  function setPlaying(p) {
    const npPlay = $('btn-play');
    if (npPlay) {
      const pi = npPlay.querySelector('.play-icon');
      const pa = npPlay.querySelector('.pause-icon');
      if (pi) pi.classList.toggle('hidden', p);
      if (pa) pa.classList.toggle('hidden', !p);
    }
    const barPlay = $('bar-play');
    if (barPlay) {
      const pi = barPlay.querySelector('.bar-play-icon');
      const pa = barPlay.querySelector('.bar-pause-icon');
      if (pi) pi.classList.toggle('hidden', p);
      if (pa) pa.classList.toggle('hidden', !p);
    }
  }

  /* Waveform */
  function drawWF() {
    const canvas = $('wf-canvas');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.parentElement.offsetWidth;
    const H = canvas.parentElement.offsetHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const seed = S.track?.seed || 42;
    const bars = Math.floor(W / 3.5);
    const mid = H / 2;
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < bars; i++) {
      const r = pr(seed + i * 7);
      const h = 3 + r * (mid - 5);
      const x = i * (W / bars);
      const w = W / bars - 1.5;
      ctx.fillStyle = `rgba(29,185,84,${0.2 + r * 0.35})`;
      ctx.beginPath();
      ctx.roundRect(x, mid - h, w, h * 2, 2);
      ctx.fill();
    }
  }

  /* Album art */
  function drawArt(canvas, seed, size) {
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d');
    const h1 = pr(seed) * 360, h2 = pr(seed + 1) * 360, h3 = pr(seed + 2) * 360;
    const g = ctx.createLinearGradient(0, 0, size, size);
    g.addColorStop(0, `hsl(${h1},60%,18%)`);
    g.addColorStop(.5, `hsl(${h2},70%,14%)`);
    g.addColorStop(1, `hsl(${h3},80%,10%)`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const cx = pr(seed + 3) * size, cy = pr(seed + 4) * size;
    const rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * .4);
    rg.addColorStop(0, 'rgba(29,185,84,0.18)');
    rg.addColorStop(1, 'transparent');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, size, size);
    const bw = size / 14;
    for (let i = 0; i < 6; i++) {
      const bh = (pr(seed + i + 10) * 0.4 + 0.1) * size;
      const bx = (i / 6) * size + bw * 0.5;
      const by = (size - bh) / 2;
      ctx.fillStyle = `rgba(255,255,255,${0.03 + pr(seed + i) * 0.04})`;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw * 0.7, bh, 3);
      ctx.fill();
    }
  }

  function pr(seed) {
    const x = Math.sin(seed + 1) * 10000;
    return x - Math.floor(x);
  }

  window.addEventListener('resize', () => { if (S.track) drawWF(); });

  /* Download / Save */
  async function downloadCurrentTrack() {
    if (!S.track) return;
    const fmt    = ($('dl-format')?.value || 'wav').toLowerCase();
    const taskId = S.track.audio_url?.split('/').pop();
    if (!taskId) return;
    if (fmt === 'wav') {
      const a = document.createElement('a');
      a.href = S.track.audio_url;
      a.download = `maz_${S.track.seed || taskId.slice(0, 8)}.wav`;
      a.click();
    } else {
      toast(`Converting to ${fmt.toUpperCase()}…`, 'info', 6000);
      try {
        const r = await fetch(`/download/${taskId}?format=${fmt}`);
        if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
        const blob = await r.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `maz_${taskId.slice(0, 8)}.${fmt}`;
        a.click();
        URL.revokeObjectURL(url);
        toast(`Downloaded as ${fmt.toUpperCase()}`, 'success');
      } catch (err) {
        toast(`Download failed: ${err.message}`, 'error', 6000);
      }
    }
    log(`Downloading (${fmt.toUpperCase()})…`);
  }

  $('btn-download').addEventListener('click', downloadCurrentTrack);
  $('bar-download').addEventListener('click', downloadCurrentTrack);

  $('btn-master')?.addEventListener('click', async () => {
    if (!S.track) return;
    const taskId = S.track.audio_url?.split('/').pop();
    if (!taskId) return;
    toast('Mastering — normalising to -14 LUFS…', 'info', 10000);
    log('Mastering track…');
    try {
      const r = await fetch(`/master/${taskId}`);
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      const blob = await r.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `maz_${taskId.slice(0, 8)}_mastered.wav`;
      a.click();
      URL.revokeObjectURL(url);
      toast('Mastered track downloaded', 'success');
      log('Mastered track ready', 'success');
    } catch (err) {
      toast(`Mastering failed: ${err.message}`, 'error', 6000);
      log(`Mastering failed: ${err.message}`, 'error');
    }
  });

  $('btn-stems')?.addEventListener('click', async () => {
    if (!S.track) return;
    const taskId = S.track.audio_url?.split('/').pop();
    if (!taskId) return;
    toast('Separating stems — this takes ~30 seconds…', 'info', 10000);
    log('Exporting stems…');
    try {
      const r = await fetch(`/stems/${taskId}`);
      if (!r.ok) {
        const txt = await r.text().catch(() => r.statusText);
        throw new Error(txt);
      }
      const blob = await r.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `maz_${taskId.slice(0, 8)}_stems.zip`;
      a.click();
      URL.revokeObjectURL(url);
      toast('Stems downloaded — vocals.wav + instrumental.wav', 'success');
      log('Stems ready', 'success');
    } catch (err) {
      toast(`Stems export failed: ${err.message}`, 'error', 6000);
      log(`Stems failed: ${err.message}`, 'error');
    }
  });

  $('btn-save').addEventListener('click', () => {
    if (!S.track) return;
    if (S.library.find(t => t.id === S.track.id)) { log('Already saved'); toast('Already in library', 'info', 2000); return; }
    S.library.unshift({ ...S.track });
    saveLib();
    $('np-badge').textContent = 'SAVED';
    log('Saved to library', 'success');
    toast('Saved to library', 'success');
  });

  /* Library — loads from server SQLite database */
  let libOffset = 0;
  const libLimit = 20;
  let libAllTracks = [];
  let libSearchTimer = null;
  let libFavOnly = false;
  let libVSOnly  = false;
  let libSort    = 'newest';
  let libView    = localStorage.getItem('lib-view') || 'grid';
  let libBulk    = false;
  let libSelected = new Set();
  let libLastClickIdx = -1;
  let _hoverAudio = null, _hoverRaf = null, _hoverTimer = null;
  function _stopHoverPreview(fillEl) {
    if (_hoverAudio) { _hoverAudio.pause(); _hoverAudio = null; }
    clearTimeout(_hoverTimer); _hoverTimer = null;
    cancelAnimationFrame(_hoverRaf); _hoverRaf = null;
    if (fillEl) fillEl.style.width = '0';
  }

  async function loadStats() {
    try {
      const r = await fetch('/history/stats');
      const d = await r.json();
      if ($('stat-total')) $('stat-total').textContent = d.total_generated || 0;
      if ($('stat-avg-bpm')) $('stat-avg-bpm').textContent = d.avg_bpm ? Math.round(d.avg_bpm) : '—';
      if ($('stat-avg-dur')) $('stat-avg-dur').textContent = d.avg_duration ? fmtDur(Math.round(d.avg_duration)) : '—';
      if ($('stat-avg-energy')) $('stat-avg-energy').textContent = d.avg_energy ? Math.round(d.avg_energy * 100) + '%' : '—';
      if ($('stat-recent')) $('stat-recent').textContent = d.recent_7_days || 0;
      if ($('stat-moods') && d.top_moods?.length) {
        $('stat-moods').innerHTML = d.top_moods.map(m =>
          `<span class="lib-mood-tag">${escHtml(m.mood)} <span style="opacity:.6">${m.n}</span></span>`
        ).join('');
      }
    } catch (e) {
      log('Stats load failed: ' + e.message, 'error');
    }
  }

  function _updateBulkBar() {
    const n = libSelected.size;
    $('lib-bulk-bar')?.classList.toggle('hidden', !libBulk || n === 0);
    const cnt = $('lib-bulk-count');
    if (cnt) cnt.textContent = `${n} selected`;
  }

  async function renderLib(search = '') {
    const g = $('lib-grid');
    g.dataset.view = libView;
    g.innerHTML = '<div class="lib-empty" style="color:var(--off)">Loading...</div>';

    try {
      const filterParam = libFavOnly ? '&favorites_only=true' : libVSOnly ? '&voice_sections_only=true' : '';
      const url = search
        ? `/history?limit=50&offset=0&sort=${libSort}${filterParam}`
        : `/history?limit=${libLimit}&offset=${libOffset}&sort=${libSort}${filterParam}`;
      const r = await fetch(url);
      const d = await r.json();

      let tracks = d.tracks;
      if (search) {
        const q = search.toLowerCase();
        tracks = tracks.filter(t =>
          (t.title || '').toLowerCase().includes(q) ||
          (t.prompt || '').toLowerCase().includes(q)
        );
      }
      if (libFavOnly)  tracks = tracks.filter(t => t.favorited);
      if (libVSOnly)   tracks = tracks.filter(t => !!t.voice_sections_path);

      libAllTracks = tracks;

      if (!tracks.length) {
        g.innerHTML = libFavOnly
          ? '<div class="lib-empty">No favorites yet — click ★ on any track to add it.</div>'
          : libVSOnly
          ? '<div class="lib-empty">No voice-sections tracks yet — apply Voice Sections to a track first.</div>'
          : '<div class="lib-empty">No tracks found.</div>';
        return;
      }

      const isGrid = libView !== 'list';
      const listHeader = isGrid ? '' : `
        <div class="lib-list-header">
          <span></span><span></span>
          <span>Title</span>
          <span>Time</span><span>BPM</span><span>Key</span><span>Mood</span>
          <span></span>
        </div>`;
      g.innerHTML = listHeader + tracks.map((t, i) => {
        const displayTitle = (t.title || t.prompt || '').slice(0, 60);
        const isTruncated  = (t.title || t.prompt || '').length > 60;
        const hasSections  = !!t.voice_sections_path;
        const audioUrl     = hasSections ? `/audio/sections_${t.job_id}` : `/audio/${t.job_id}`;
        const checked      = libSelected.has(t.job_id) ? 'checked' : '';
        // List rows must keep display intact for grid flow; use visibility:hidden instead of display:none
        const cbClass      = libBulk ? 'lib-cb-wrap' : isGrid ? 'lib-cb-wrap hidden' : 'lib-cb-wrap lib-cb-invis';
        const favFill      = t.favorited ? 'currentColor' : 'none';
        const favTitle     = t.favorited ? 'Remove from favorites' : 'Add to favorites';
        const favSvg       = `<svg viewBox="0 0 16 16" width="13" height="13" fill="${favFill}" stroke="currentColor" stroke-width="1.4"><path d="M8 1.5l1.8 3.6 4 .6-2.9 2.8.7 4L8 10.4l-3.6 1.9.7-4L2.2 5.7l4-.6z"/></svg>`;

        if (isGrid) return `
        <div class="lib-card" data-audio-url="${audioUrl}" data-jobid="${t.job_id}" data-idx="${i}">
          <label class="${cbClass}"><input type="checkbox" class="lib-cb" ${checked} data-jobid="${t.job_id}" /></label>
          <canvas class="lib-card-art" width="180" height="180" data-seed="${t.seed || 42}"></canvas>
          <div class="lib-card-title-row">
            <div class="lib-card-title" data-jobtitle="${escHtml(t.title || '')}" data-jobprompt="${escHtml(t.prompt || '')}">${escHtml(displayTitle)}${isTruncated ? '…' : ''}</div>
            ${hasSections ? '<span class="lib-vs-badge" title="Voice sections applied">VS</span>' : ''}
            <button class="lib-rename-btn" data-a="rename" data-jobid="${t.job_id}" title="Rename">✎</button>
          </div>
          <div class="lib-card-meta">
            ${fmtDur(t.duration)} · ${new Date(t.created_at).toLocaleDateString('en', { month: 'short', day: 'numeric' })}
            ${t.bpm ? ` · ${Math.round(t.bpm)} BPM` : ''}
            ${t.key ? ` · ${t.key}` : ''}
            ${t.mood ? ` · ${t.mood}` : ''}
          </div>
          <div class="lib-card-actions">
            <button class="lib-btn" data-a="play" data-jobid="${t.job_id}" data-audio-url="${audioUrl}" data-seed="${t.seed || 42}" data-dur="${t.duration > 0 ? t.duration : 0}" data-prompt="${escHtml(t.prompt || '')}" data-lyrics="${escHtml(t.lyrics || '')}">Play</button>
            <button class="lib-btn" data-a="dl" data-jobid="${t.job_id}" data-audio-url="${audioUrl}">DL</button>
            <button class="lib-btn" data-a="stems" data-jobid="${t.job_id}" title="Export vocals + instrumental stems">Stems</button>
            <button class="lib-btn lib-fav-btn${t.favorited ? ' active' : ''}" data-a="fav" data-jobid="${t.job_id}" data-fav="${t.favorited ? '1' : '0'}" title="${favTitle}">${favSvg}</button>
            <button class="lib-btn del" data-a="del" data-jobid="${t.job_id}">X</button>
          </div>
          <div class="lib-card-preview-bar"><div class="lib-card-preview-fill"></div></div>
        </div>`;

        return `
        <div class="lib-card lib-card--list" data-audio-url="${audioUrl}" data-jobid="${t.job_id}" data-idx="${i}">
          <label class="${cbClass}"><input type="checkbox" class="lib-cb" ${checked} data-jobid="${t.job_id}" /></label>
          <canvas class="lib-card-art-sm" width="52" height="52" data-seed="${t.seed || 42}"></canvas>
          <div class="lib-list-info">
            <div class="lib-list-title" data-jobtitle="${escHtml(t.title || '')}" data-jobprompt="${escHtml(t.prompt || '')}">${escHtml(displayTitle)}${isTruncated ? '…' : ''}${hasSections ? ' <span class="lib-vs-badge">VS</span>' : ''}</div>
            <div class="lib-list-sub">${new Date(t.created_at).toLocaleDateString('en', { month: 'short', day: 'numeric', year: 'numeric' })}</div>
          </div>
          <span class="lib-list-col">${fmtDur(t.duration)}</span>
          <span class="lib-list-col">${t.bpm ? Math.round(t.bpm) : '—'}</span>
          <span class="lib-list-col">${t.key || '—'}</span>
          <span class="lib-list-col lib-list-mood-col">${t.mood || '—'}</span>
          <div class="lib-list-actions">
            <button class="lib-btn" data-a="play" data-jobid="${t.job_id}" data-audio-url="${audioUrl}" data-seed="${t.seed || 42}" data-dur="${t.duration > 0 ? t.duration : 0}" data-prompt="${escHtml(t.prompt || '')}" data-lyrics="${escHtml(t.lyrics || '')}">▶</button>
            <button class="lib-btn" data-a="dl" data-jobid="${t.job_id}" data-audio-url="${audioUrl}">↓</button>
            <button class="lib-btn" data-a="stems" data-jobid="${t.job_id}" title="Export stems">⊜</button>
            <button class="lib-btn lib-fav-btn${t.favorited ? ' active' : ''}" data-a="fav" data-jobid="${t.job_id}" data-fav="${t.favorited ? '1' : '0'}" title="${favTitle}">${favSvg}</button>
            <button class="lib-btn del" data-a="del" data-jobid="${t.job_id}">✕</button>
          </div>
        </div>`;
      }).join('');

      // Draw art canvases
      g.querySelectorAll('canvas[data-seed]').forEach(c => drawArt(c, parseInt(c.dataset.seed), isGrid ? 180 : 52));

      // Hover audio preview — both grid and list
      g.querySelectorAll('.lib-card[data-audio-url]').forEach(card => {
        const fill = card.querySelector('.lib-card-preview-fill'); // null in list view — handled gracefully
        card.addEventListener('mouseenter', () => {
          _stopHoverPreview(fill);
          _hoverAudio = new Audio(card.dataset.audioUrl);
          _hoverAudio.volume = 0.55;
          _hoverAudio.play().catch(() => {});
          const t0 = Date.now();
          const PREVIEW_MS = 5000;
          function tick() {
            if (!_hoverAudio) return;
            if (fill) fill.style.width = Math.min((Date.now() - t0) / PREVIEW_MS * 100, 100) + '%';
            _hoverRaf = requestAnimationFrame(tick);
          }
          tick();
          _hoverTimer = setTimeout(() => _stopHoverPreview(fill), PREVIEW_MS);
        });
        card.addEventListener('mouseleave', () => _stopHoverPreview(fill));
      });

      // Bulk checkboxes — shift-click range select
      g.querySelectorAll('.lib-cb').forEach(cb => {
        cb.addEventListener('change', e => {
          const jobId = cb.dataset.jobid;
          const card  = cb.closest('.lib-card');
          const idx   = parseInt(card.dataset.idx);
          if (e.shiftKey && libLastClickIdx >= 0) {
            const lo = Math.min(libLastClickIdx, idx);
            const hi = Math.max(libLastClickIdx, idx);
            [...g.querySelectorAll('.lib-card')].forEach((c, k) => {
              if (k < lo || k > hi) return;
              const jid = c.dataset.jobid;
              const box = c.querySelector('.lib-cb');
              if (cb.checked) { libSelected.add(jid);    if (box) box.checked = true; }
              else             { libSelected.delete(jid); if (box) box.checked = false; }
            });
          } else {
            if (cb.checked) libSelected.add(jobId);
            else            libSelected.delete(jobId);
          }
          libLastClickIdx = idx;
          _updateBulkBar();
        });
      });

      // Load more button
      const lm = $('lib-load-more');
      if (lm) lm.classList.toggle('hidden', !(!search && d.total > libOffset + libLimit));

      // Action buttons (play / dl / fav / rename / del)
      g.querySelectorAll('[data-a]').forEach(btn => {
        btn.addEventListener('click', async e => {
          e.stopPropagation();
          const jobId  = btn.dataset.jobid;
          const action = btn.dataset.a;

          if (action === 'play') {
            const card = btn.closest('.lib-card');
            const titleEl = card?.querySelector('.lib-card-title, .lib-list-title');
            const track = {
              id: Date.now(),
              job_id: jobId,
              title: titleEl?.dataset.jobtitle || '',
              prompt: btn.dataset.prompt,
              lyrics: btn.dataset.lyrics || null,
              duration: parseFloat(btn.dataset.dur),
              seed: parseInt(btn.dataset.seed),
              audio_url: btn.dataset.audioUrl || `/audio/${jobId}`,
              created: new Date().toISOString(),
            };
            S.track = track; loadTrack(track);
            $$('.sidebar-item').forEach(b => b.classList.remove('active'));
            $$('.page').forEach(p => p.classList.remove('active'));
            document.querySelector('[data-view="studio"]').classList.add('active');
            $('page-studio').classList.add('active');
            const tryPlay = () => {
              audio.play()
                .then(() => setPlaying(true))
                .catch(err => { if (err.name !== 'AbortError') { setPlaying(false); } });
            };
            if (audio.readyState >= 3) {
              tryPlay();
            } else {
              audio.addEventListener('canplay', tryPlay, { once: true });
            }

          } else if (action === 'dl') {
            const a = document.createElement('a');
            a.href = btn.dataset.audioUrl || `/audio/${jobId}`;
            a.download = `maz_${jobId.slice(0, 8)}.wav`;
            a.click();

          } else if (action === 'stems') {
            toast('Separating stems — this takes ~30 seconds…', 'info', 10000);
            try {
              const r = await fetch(`/stems/${jobId}`);
              if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
              const blob = await r.blob();
              const url  = URL.createObjectURL(blob);
              const a    = document.createElement('a');
              a.href     = url;
              a.download = `maz_${jobId.slice(0, 8)}_stems.zip`;
              a.click();
              URL.revokeObjectURL(url);
              toast('Stems downloaded — vocals.wav + instrumental.wav', 'success');
            } catch (err) {
              toast(`Stems failed: ${err.message}`, 'error', 6000);
            }

          } else if (action === 'fav') {
            try {
              const res  = await fetch(`/history/${jobId}/favorite`, { method: 'PATCH' });
              const data = await res.json();
              const isFav = data.favorited;
              btn.dataset.fav = isFav ? '1' : '0';
              btn.title = isFav ? 'Remove from favorites' : 'Add to favorites';
              btn.classList.toggle('active', isFav);
              btn.querySelector('svg').setAttribute('fill', isFav ? 'currentColor' : 'none');
              if (libFavOnly && !isFav) btn.closest('.lib-card').remove();
            } catch (e) {
              log('Favorite toggle failed: ' + e.message, 'error');
              toast('Favorite toggle failed', 'error');
            }

          } else if (action === 'del') {
            if (!confirm('Remove this track from history?')) return;
            try {
              await fetch(`/history/${jobId}`, { method: 'DELETE' });
              renderLib(($('lib-search') || {}).value || '');
              loadStats();
            } catch (e) {
              log('Delete failed: ' + e.message, 'error');
              toast('Delete failed', 'error');
            }

          } else if (action === 'rename') {
            const card      = btn.closest('.lib-card');
            const titleEl   = card.querySelector('.lib-card-title');
            const currentTitle = titleEl.dataset.jobtitle || titleEl.dataset.jobprompt || '';
            const input     = document.createElement('input');
            input.type      = 'text';
            input.className = 'lib-rename-input';
            input.value     = currentTitle;
            input.maxLength = 200;
            titleEl.replaceWith(input);
            btn.style.display = 'none';
            input.focus();
            input.select();

            let committed = false;
            async function commitRename() {
              if (committed) return;
              committed = true;
              const newTitle = input.value.trim();
              const restored = document.createElement('div');
              restored.className          = 'lib-card-title';
              restored.dataset.jobtitle   = newTitle;
              restored.dataset.jobprompt  = titleEl.dataset.jobprompt;
              const display = (newTitle || titleEl.dataset.jobprompt || '').slice(0, 60);
              restored.textContent = display + ((newTitle || titleEl.dataset.jobprompt || '').length > 60 ? '…' : '');
              input.replaceWith(restored);
              btn.style.display = '';

              if (newTitle === currentTitle) return;
              try {
                await fetch(`/history/${jobId}/title`, {
                  method: 'PATCH',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ title: newTitle }),
                });
                log(`Renamed to "${newTitle || '(cleared)'}"`, 'success');
                toast('Renamed', 'success', 2000);
              } catch (err) {
                log('Rename failed: ' + err.message, 'error');
                toast('Rename failed', 'error');
                restored.dataset.jobtitle = currentTitle;
                restored.textContent = currentTitle.slice(0, 60) + (currentTitle.length > 60 ? '…' : '');
              }
            }

            input.addEventListener('keydown', e => {
              if (e.key === 'Enter')  { e.preventDefault(); commitRename(); }
              if (e.key === 'Escape') { input.value = currentTitle; commitRename(); }
            });
            input.addEventListener('blur', commitRename, { once: true });
          }
        });
      });

    } catch (e) {
      g.innerHTML = `<div class="lib-empty">Error loading history: ${escHtml(e.message)}</div>`;
    }
  }

  // Library filter tabs — All / Favorites / Voice Sections
  ['lib-tab-all', 'lib-tab-favs', 'lib-tab-vs'].forEach(id => {
    $(id)?.addEventListener('click', () => {
      libFavOnly = id === 'lib-tab-favs';
      libVSOnly  = id === 'lib-tab-vs';
      $('lib-tab-all')?.classList.toggle('active',  id === 'lib-tab-all');
      $('lib-tab-favs')?.classList.toggle('active', id === 'lib-tab-favs');
      $('lib-tab-vs')?.classList.toggle('active',   id === 'lib-tab-vs');
      libOffset = 0;
      renderLib(($('lib-search') || {}).value?.trim() || '');
    });
  });

  // Search
  $('lib-search')?.addEventListener('input', () => {
    clearTimeout(libSearchTimer);
    libSearchTimer = setTimeout(() => {
      renderLib($('lib-search').value.trim());
    }, 400);
  });

  // Refresh button
  $('btn-refresh-lib')?.addEventListener('click', () => {
    libOffset = 0;
    renderLib();
    loadStats();
  });

  // Export CSV
  $('btn-export-csv')?.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = '/history/export';
    a.download = 'maz_history.csv';
    a.click();
    log(t('log_downloading'), 'success');
    toast('Downloading...', 'info', 2000);
  });

  // Load more
  $('btn-load-more')?.addEventListener('click', () => {
    libOffset += libLimit;
    renderLib();
  });

  // Sort pills
  document.querySelectorAll('.lib-sort-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      libSort = pill.dataset.sort;
      libOffset = 0;
      document.querySelectorAll('.lib-sort-pill').forEach(p => p.classList.toggle('active', p === pill));
      renderLib(($('lib-search') || {}).value?.trim() || '');
    });
  });

  // Grid / list view toggle
  function _applyView(v) {
    libView = v;
    localStorage.setItem('lib-view', v);
    $('btn-view-grid')?.classList.toggle('active', v === 'grid');
    $('btn-view-list')?.classList.toggle('active', v === 'list');
  }
  _applyView(libView); // apply saved pref on load
  $('btn-view-grid')?.addEventListener('click', () => { _applyView('grid'); renderLib(($('lib-search') || {}).value?.trim() || ''); });
  $('btn-view-list')?.addEventListener('click', () => { _applyView('list'); renderLib(($('lib-search') || {}).value?.trim() || ''); });

  // Bulk select toggle
  $('btn-bulk-toggle')?.addEventListener('click', () => {
    libBulk = !libBulk;
    libSelected.clear();
    libLastClickIdx = -1;
    $('btn-bulk-toggle')?.classList.toggle('active', libBulk);
    $('btn-bulk-toggle').textContent = libBulk ? 'Cancel' : 'Select';
    _updateBulkBar();
    // Show/hide checkboxes without full re-render (grid: display:none / list: visibility:hidden)
    document.querySelectorAll('.lib-cb-wrap').forEach(w => {
      if (w.closest('.lib-card--list')) {
        w.classList.toggle('lib-cb-invis', !libBulk);
        w.classList.remove('hidden');
      } else {
        w.classList.toggle('hidden', !libBulk);
        w.classList.remove('lib-cb-invis');
      }
    });
  });

  // Bulk: download selected (sequential anchor clicks)
  $('btn-bulk-dl')?.addEventListener('click', () => {
    if (!libSelected.size) return;
    [...libSelected].forEach((jobId, i) => {
      setTimeout(() => {
        const a = document.createElement('a');
        a.href = `/audio/${jobId}`;
        a.download = `maz_${jobId.slice(0, 8)}.wav`;
        a.click();
      }, i * 300);
    });
    toast(`Downloading ${libSelected.size} track${libSelected.size > 1 ? 's' : ''}`, 'info', 3000);
  });

  // Bulk: export ZIP
  $('btn-bulk-zip')?.addEventListener('click', async () => {
    if (!libSelected.size) return;
    const ids = [...libSelected].join(',');
    const a = document.createElement('a');
    a.href = `/history/bulk-download?job_ids=${encodeURIComponent(ids)}`;
    a.download = 'maz-export.zip';
    a.click();
    toast(`Exporting ${libSelected.size} track${libSelected.size > 1 ? 's' : ''} as ZIP`, 'info', 3000);
  });

  // Bulk: delete selected
  $('btn-bulk-del')?.addEventListener('click', async () => {
    if (!libSelected.size) return;
    if (!confirm(`Delete ${libSelected.size} track${libSelected.size > 1 ? 's' : ''}? This cannot be undone.`)) return;
    const ids = [...libSelected];
    let failed = 0;
    await Promise.all(ids.map(async jobId => {
      try { await fetch(`/history/${jobId}`, { method: 'DELETE' }); }
      catch (_) { failed++; }
    }));
    libSelected.clear();
    libBulk = false;
    $('btn-bulk-toggle').textContent = 'Select';
    $('btn-bulk-toggle')?.classList.remove('active');
    _updateBulkBar();
    renderLib(($('lib-search') || {}).value?.trim() || '');
    loadStats();
    if (failed) toast(`${failed} deletion${failed > 1 ? 's' : ''} failed`, 'error');
    else toast(`Deleted ${ids.length} track${ids.length > 1 ? 's' : ''}`, 'success');
  });

  // Bulk: clear selection
  $('btn-bulk-deselect')?.addEventListener('click', () => {
    libSelected.clear();
    libLastClickIdx = -1;
    document.querySelectorAll('.lib-cb').forEach(cb => { cb.checked = false; });
    _updateBulkBar();
  });

  // Keep localStorage save for current session track
  $('btn-save')?.addEventListener('click', () => {
    // Already handled above — DB saves automatically on generation
    // This just updates the badge
  });
  function saveLib() { /* no-op — DB handles persistence */ }

  /* Log */
  function log(msg, type = '') {
    const el = document.createElement('div');
    el.className = `log-line${type ? ` log--${type}` : ''}`;
    const ts = new Date().toLocaleTimeString('en', { hour12: false });
    el.textContent = `[${ts}] ${msg}`;
    $('log-feed').appendChild(el);
    $('log-feed').scrollTop = $('log-feed').scrollHeight;
    while ($('log-feed').children.length > 80) $('log-feed').removeChild($('log-feed').firstChild);
  }

  /* Browser push notifications */
  function maybeNotify(title, body) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    try {
      const n = new Notification(title, { body, silent: false });
      n.onclick = () => { window.focus(); n.close(); };
    } catch (_) {}
  }
  function requestNotifPermission() {
    if (!('Notification' in window) || Notification.permission !== 'default') return;
    Notification.requestPermission();
  }

  /* Toast notifications */
  const TOAST_ICONS = { success: '✓', error: '✕', warn: '⚠', info: 'ℹ' };
  function toast(msg, type = 'info', durationMs = 3500) {
    const container = $('toast-container');
    const t = document.createElement('div');
    t.className = `toast toast--${type}`;
    t.innerHTML = `
      <span class="toast-icon">${TOAST_ICONS[type] || 'ℹ'}</span>
      <span class="toast-body"><span class="toast-msg">${escHtml(String(msg))}</span></span>
      <button class="toast-close" aria-label="Dismiss">✕</button>
      <div class="toast-progress" style="animation-duration:${durationMs}ms"></div>`;
    container.appendChild(t);
    t.querySelector('.toast-close').addEventListener('click', () => dismiss(t));
    const tid = setTimeout(() => dismiss(t), durationMs);
    t._tid = tid;
    function dismiss(el) {
      clearTimeout(el._tid);
      el.classList.add('toast--out');
      el.addEventListener('animationend', () => el.remove(), { once: true });
    }
  }
  $('btn-clear-log').addEventListener('click', () => {
    $('log-feed').innerHTML = '<div class="log-line log--dim">Cleared.</div>';
  });

  /* Keyboard shortcuts */
  document.addEventListener('keydown', e => {
    const inInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
    const mod     = e.ctrlKey || e.metaKey;

    // Ctrl/Cmd+D — download current track
    if (mod && e.key === 'd') {
      e.preventDefault();
      if (S.track) downloadCurrentTrack();
      return;
    }
    // Escape — close open modal
    if (e.key === 'Escape') {
      const modal = $('ai-modal');
      if (modal && !modal.classList.contains('hidden')) {
        modal.classList.add('hidden');
        return;
      }
    }
    if (inInput || mod) return;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        if (S.track) {
          const paused = audio.paused;
          paused ? audio.play().then(() => setPlaying(true)) : (audio.pause(), setPlaying(false));
        }
        break;
      case 'g':
      case 'G':
        // Go to Studio + focus prompt
        $$('.sidebar-item').forEach(b => b.classList.remove('active'));
        $$('.page').forEach(p => p.classList.remove('active'));
        document.querySelector('[data-view="studio"]').classList.add('active');
        $('page-studio').classList.add('active');
        $('prompt')?.focus();
        break;
      case 'l':
      case 'L':
        // Go to Library
        document.querySelector('[data-view="library"]')?.click();
        break;
      case 'f':
      case 'F':
        // Favourite the active track (if loaded)
        if (S.track?.job_id) {
          fetch(`/history/${S.track.job_id}/favorite`, { method: 'PATCH' })
            .then(r => r.json())
            .then(d => toast(d.favorited ? 'Added to favorites' : 'Removed from favorites', 'success', 2000))
            .catch(() => {});
        }
        break;
      case 'ArrowRight':
        if (S.track && audio.duration) audio.currentTime = Math.min(audio.duration, audio.currentTime + 10);
        break;
      case 'ArrowLeft':
        if (S.track) audio.currentTime = Math.max(0, audio.currentTime - 10);
        break;
    }
  });

  /* Helpers */
  function fmt(s) {
    if (!s || isNaN(s)) return '0:00';
    return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
  }
  function escHtml(str) {
    return String(str).replace(/[&<>"']/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }


  /* ══ EVALUATION PAGE ══════════════════════════════════════ */
  const E = {
    running: false,
    audios: [new Audio(), new Audio(), new Audio()],
    jobIds: [null, null, null],
    taskIds: [null, null, null],
    genTimes: [null, null, null],
  };

  // Stop all eval audios when main audio plays and vice versa
  audio.addEventListener('play', () => E.audios.forEach(a => a.pause()));
  E.audios.forEach((a, i) => {
    a.addEventListener('play', () => {
      audio.pause();
      E.audios.forEach((b, j) => { if (j !== i) b.pause(); });
    });
    a.addEventListener('timeupdate', () => {
      const pct = a.duration ? (a.currentTime / a.duration) * 100 : 0;
      const fill = $(`eval-wf-fill-${i}`);
      const time = $(`eval-time-${i}`);
      if (fill) fill.style.width = pct + '%';
      if (time) time.textContent = fmt(a.currentTime);
    });
    a.addEventListener('ended', () => {
      const btn = $(`eval-play-${i}`);
      if (btn) btn.innerHTML = '<svg viewBox="0 0 20 20"><polygon points="5,3 18,10 5,17" fill="currentColor"/></svg>';
    });
  });

  // Play buttons
  [0, 1, 2].forEach(i => {
    $(`eval-play-${i}`)?.addEventListener('click', () => {
      const a = E.audios[i];
      if (!a.src) return;
      if (a.paused) {
        a.play();
        $(`eval-play-${i}`).innerHTML = '<svg viewBox="0 0 20 20"><rect x="4" y="3" width="4" height="14" rx="1" fill="currentColor"/><rect x="12" y="3" width="4" height="14" rx="1" fill="currentColor"/></svg>';
      } else {
        a.pause();
        $(`eval-play-${i}`).innerHTML = '<svg viewBox="0 0 20 20"><polygon points="5,3 18,10 5,17" fill="currentColor"/></svg>';
      }
    });

    $(`eval-wf-${i}`)?.addEventListener('click', e => {
      const a = E.audios[i];
      if (!a.duration) return;
      const r = $(`eval-wf-${i}`).getBoundingClientRect();
      a.currentTime = ((e.clientX - r.left) / r.width) * a.duration;
    });
  });

  // Draw eval waveform
  function drawEvalWF(idx, seed) {
    const canvas = $(`eval-wf-canvas-${idx}`);
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.parentElement.offsetWidth;
    const H = canvas.parentElement.offsetHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const bars = Math.floor(W / 3.5);
    const mid = H / 2;
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < bars; i++) {
      const r = pr(seed + i * 7);
      const h = 2 + r * (mid - 3);
      ctx.fillStyle = `rgba(29,185,84,${0.2 + r * 0.35})`;
      ctx.beginPath();
      ctx.roundRect(i * (W / bars), mid - h, W / bars - 1.5, h * 2, 2);
      ctx.fill();
    }
  }

  // Single variant generation
  // seed: shared across all variants so only quality settings differ
  async function generateVariant(idx, seed) {
    const variant = $$('.eval-variant')[idx];
    const steps    = parseInt(variant.querySelector('.eval-steps').value);
    const duration = parseInt(variant.querySelector('.eval-duration').value);
    const guidance = parseFloat(variant.querySelector('.eval-guidance').value);
    const prompt   = $('eval-prompt').value.trim();
    const lyrics   = $('eval-lyrics').value.trim() || null;

    variant.classList.add('active');
    $(`eval-result-${idx}`).classList.remove('hidden');
    $(`eval-status-${idx}`).textContent = t('eval_generating');
    $(`eval-meta-${idx}`).textContent = `${steps} steps · ${duration}s`;

    const startTime = Date.now();

    try {
      const res = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt, lyrics, duration,
          guidance_scale: guidance,
          infer_steps: steps,
          seed: seed ?? -1,
          bpm: null,
          key: null,
        }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || res.statusText);
      }
      const data = await res.json();
      const jobId = data.job_id;
      E.jobIds[idx] = jobId;

      // Wait for completion using both WS (fast path) and HTTP polling (reliable fallback).
      // The WS handler is attached to S.ws at submit time — if the socket reconnects
      // mid-generation S.ws is reassigned, so the HTTP poll is the only reliable path.
      await new Promise((resolve, reject) => {
        let settled = false;
        function settle(fn) {
          if (settled) return;
          settled = true;
          clearTimeout(timeoutId);
          clearInterval(pollId);
          if (wsTarget) wsTarget.removeEventListener('message', wsHandler);
          fn();
        }

        const timeoutId = setTimeout(
          () => settle(() => reject(new Error('Timed out after 600s'))),
          600_000
        );

        // HTTP poll every 5s — works through WS reconnects
        const pollId = setInterval(async () => {
          try {
            const r = await fetch(`/job/${jobId}`);
            if (!r.ok) return;
            const d = await r.json();
            if (d.status === 'completed' && d.audio_url) {
              E.taskIds[idx] = d.audio_url.split('/').pop();
              settle(() => resolve(d));
            } else if (d.status === 'failed') {
              settle(() => reject(new Error(d.error || 'Generation failed')));
            }
          } catch (_) {}
        }, 5000);

        // WS fast path — instant notification while socket is alive
        const wsTarget = (S.ws && S.ws.readyState === WebSocket.OPEN) ? S.ws : null;
        const wsHandler = (event) => {
          let msg;
          try { msg = JSON.parse(event.data); } catch { return; }
          if (msg.type === 'job_completed' && msg.job_id === jobId) {
            E.taskIds[idx] = msg.audio_url.split('/').pop();
            settle(() => resolve(msg));
          }
          if (msg.type === 'job_failed' && msg.job_id === jobId) {
            settle(() => reject(new Error(msg.error)));
          }
        };
        if (wsTarget) wsTarget.addEventListener('message', wsHandler);
      });

      const genTime = ((Date.now() - startTime) / 1000).toFixed(1);
      E.genTimes[idx] = genTime;

      const audioUrl = `/audio/${E.taskIds[idx]}`;
      E.audios[idx].src = audioUrl;
      E.audios[idx].load();
      E.audios[idx].addEventListener('loadedmetadata', () => {
        drawEvalWF(idx, idx * 1000 + steps);
      }, { once: true });

      drawArt($(`eval-art-${idx}`), idx * 1000 + steps, 48);

      $(`eval-status-${idx}`).textContent = t('eval_ready');
      $(`eval-meta-${idx}`).textContent = `${steps} steps · ${duration}s · ${genTime}s gen time`;

      try {
        const ar = await fetch(`/analyze/${E.taskIds[idx]}`);
        if (ar.ok) {
          const a = await ar.json();
          $(`eval-bpm-${idx}`).textContent = Math.round(a.bpm);
          $(`eval-key-${idx}`).textContent = a.key;
          $(`eval-mood-${idx}`).textContent = a.mood;
          $(`eval-energy-${idx}`).textContent = Math.round(a.energy * 100) + '%';
          $(`eval-gentime-${idx}`).textContent = genTime + 's';
          $(`eval-analysis-${idx}`).classList.remove('hidden');
        }
      } catch { }

      variant.classList.remove('active');
      return true;

    } catch (err) {
      $(`eval-status-${idx}`).textContent = t('eval_failed') + ': ' + err.message;
      variant.classList.remove('active');
      return false;
    }
  }

  // Run all variants sequentially
  $('btn-eval-run')?.addEventListener('click', async () => {
    const prompt = $('eval-prompt').value.trim();
    if (!prompt) { log('Enter a prompt for evaluation', 'warn'); return; }
    if (E.running) return;

    if (S.jobId) {
      log('Wait for Studio generation to finish first', 'warn');
      return;
    }

    E.running = true;
    $('btn-eval-run').disabled = true;
    $('eval-run-idle').classList.add('hidden');
    $('eval-run-busy').classList.remove('hidden');
    $('eval-progress').classList.remove('hidden');

    // Shared seed: all variants use the same seed so only quality settings (steps/guidance) differ.
    // This makes the comparison meaningful — the songs are the same generation, just at different quality.
    const evalSeed = Math.floor(Math.random() * (2 ** 31 - 1));
    log(`Starting evaluation (seed ${evalSeed}) — generating 3 variants sequentially`, 'ai');

    for (let i = 0; i < 3; i++) {
      const evalRunLabel = $('eval-run-label') || $('eval-progress-label');
      if (evalRunLabel) evalRunLabel.textContent = `Generating variant ${i + 1} of 3...`;
      $('eval-progress-fill').style.width = `${(i / 3) * 100}%`;
      const ok = await generateVariant(i, evalSeed);
      if (!ok) {
        log(`Variant ${i + 1} failed`, 'error');
      } else {
        log(`Variant ${i + 1} complete`, 'success');
      }
    }

    $('eval-progress-fill').style.width = '100%';
    const evalRunLabelFinal = $('eval-run-label') || $('eval-progress-label');
    if (evalRunLabelFinal) evalRunLabelFinal.textContent = 'All variants complete!';
    E.running = false;
    $('btn-eval-run').disabled = false;
    $('eval-run-idle').classList.remove('hidden');
    $('eval-run-busy').classList.add('hidden');
    log('Evaluation complete — compare the results above', 'success');
  });



  /* ══ INTERNATIONALIZATION (EN / KK / RU) ════════════════════ */
  const TRANSLATIONS = {
    en: {
      nav_studio: 'Studio',
      nav_library: 'My Library',
      nav_evaluate: 'Evaluate',
      nav_voice: 'Voice',
      nav_train: 'Train Voice',
      studio_title: 'Create',
      studio_sub: 'Describe your track, add lyrics, hit generate',
      label_description: 'Music description',
      btn_enhance: 'Enhance with AI',
      btn_write_ai: 'Write with AI',
      label_lyrics: 'Lyrics',
      label_optional: 'optional',
      label_optional_parens: '(optional)',
      label_style: 'Style',
      style_hint: 'tap to add',
      theme_light: 'Light mode',
      theme_dark: 'Dark mode',
      skin_label: 'Skin',
      skin_default: 'Default',
      skin_amber: 'Amber',
      skin_paper: 'Paper',
      label_parameters: 'Parameters',
      auto_duration: 'Auto',
      param_duration: 'Duration',
      param_guidance: 'Guidance',
      param_steps: 'Steps',
      param_bpm: 'BPM',
      param_key: 'Key',
      param_seed: 'Seed',
      btn_generate: 'Generate Track',
      btn_cancel: 'Cancel',
      player_empty_title: 'Nothing playing',
      player_empty_sub: 'Your generated track will appear here',
      btn_download: 'Download',
      btn_save: 'Save',
      queue_title: 'Queue',
      activity_title: 'Activity',
      btn_clear: 'Clear',
      library_title: 'Your Library',
      eval_title: 'Evaluation',
      btn_run_comparison: 'Run Comparison',
      badge_speed: 'Speed',
      badge_recommended: 'Recommended',
      badge_quality: 'Best Quality',
      badge_new: 'NEW',
      unit_sec: 'sec',
      range_creative: 'creative',
      range_strict: 'strict',
      range_fast: 'fast',
      range_quality: 'quality',
      status_online: 'Online',
      status_offline: 'Offline',
      status_connecting: 'Connecting…',
      queue_idle: 'No active generation',
      queue_waiting: 'waiting',
      log_connected: 'Connected',
      log_started: 'Generation started',
      log_ready: 'Track ready',
      log_ready_init: 'Ready.',
      log_saved: 'Saved to library',
      log_downloading: 'Downloading...',
      log_prompt_applied: 'Prompt applied',
      log_lyrics_applied: 'Lyrics applied',
      stat_total: 'Total Generated',
      stat_avg_bpm: 'Avg BPM',
      stat_avg_dur: 'Avg Duration',
      stat_avg_energy: 'Avg Energy',
      stat_last7: 'Last 7 Days',
      stat_top_moods: 'Top Moods',
      lib_history: 'Track History',
      lib_search_placeholder: 'Search prompts...',
      lib_empty: 'No tracks yet. Generate something first.',
      btn_refresh: 'Refresh',
      btn_load_more: 'Load more',
      btn_export_csv: 'Export CSV',
      eval_sub: 'Compare the same prompt at different quality settings',
      eval_prompt_label: 'Prompt to evaluate',
      eval_presets_title: 'Comparison Presets',
      eval_presets_hint: 'customize each variant',
      eval_variant_a: 'Variant A — Fast',
      eval_variant_b: 'Variant B — Balanced',
      eval_variant_c: 'Variant C — Quality',
      // Vocals selector
      label_vocals: 'Vocals',
      vocal_auto: 'Auto',
      vocal_female: 'Female',
      vocal_male: 'Male',
      vocal_mixed: 'Mixed',
      // UI controls
      btn_history: 'History',
      badge_idle: 'idle',
      badge_active: 'active',
      btn_stems: 'Stems',
      btn_normalize: 'Normalize',
      vs_hint: 'Assign a different voice to each section of the song.',
      vs_processing: 'Processing…',
      btn_apply_sections: 'Apply Voice Sections',
      lib_tab_all: 'All',
      lib_tab_favs: 'Favorites',
      lib_tab_vs: 'Voice Sections',
      lib_sort_label: 'Sort:',
      sort_newest: 'Newest',
      sort_oldest: 'Oldest',
      btn_select: 'Select',
      btn_bulk_zip: 'Export ZIP',
      btn_bulk_del: 'Delete',
      metric_gen_time: 'Gen Time',
      eval_generating: 'Generating…',
      eval_ready: 'Ready',
      eval_failed: 'Failed',
      // Voice toggle
      voice_toggle_label: 'Sing in my voice',
      // Video panel
      video_no_track: 'No track yet',
      video_no_track_sub: 'Generate a music track first, then come back here to make a video.',
      video_title: 'AI Video Generation',
      video_sub: 'Wan2.1 will generate a cinematic clip for each lyric section, then stitch them with your audio.',
      video_generating: 'Generating Video',
      btn_gen_video: 'Generate Video',
      btn_dl_mp4: 'Download MP4',
      btn_play_audio: 'Play with Audio',
      btn_regenerate: 'Regenerate',
      // Analysis
      analyzing_audio: 'Analyzing audio…',
      metric_bpm: 'BPM',
      metric_key: 'Key',
      metric_mood: 'Mood',
      metric_energy: 'Energy',
      spectrum_bass: 'Bass',
      spectrum_mid: 'Mid',
      spectrum_treble: 'Treble',
      // Voice page
      voice_title: 'My Voice',
      voice_sub: 'Save a voice profile — then use it in Studio to sing any song in your voice',
      voice_svc_sub: 'Zero-shot · any recording',
      voice_rvc_label: 'RVC Model',
      voice_rvc_sub: 'Upload trained model',
      voice_step_record: 'Record or upload a 10–60 second voice sample',
      btn_start_record: 'Start Recording',
      btn_stop_record: 'Stop Recording',
      voice_or: 'or',
      btn_upload_audio: 'Upload Audio',
      btn_save_profile: 'Save Profile',
      voice_isolating: 'Isolating vocals',
      voice_denoising: 'Denoising',
      voice_normalizing: 'Normalizing',
      voice_rvc_step: 'Upload your trained RVC model file',
      btn_upload_pth: 'Upload .pth Model',
      btn_upload_index: 'Upload .index',
      btn_save_rvc: 'Save RVC Model',
      voice_saved: 'Saved voices',
      voice_empty: 'No voices saved yet — record or upload one above.',
      // Train page
      train_title: 'Train Voice Model',
      train_sub: 'Upload multiple voice recordings to train a custom speaker model',
      train_data_label: 'Training Data',
      train_drop_title: 'Drop audio files here or',
      train_drop_link: 'click to browse',
      train_drop_hint: 'WAV · MP3 · FLAC · Multiple files recommended (5–60 seconds each)',
      train_params_label: 'Training Parameters',
      train_model_name: 'Model Name',
      train_epochs: 'Epochs',
      train_epochs_hint: 'More = better quality, slower',
      train_sample_rate: 'Sample Rate',
      train_batch_size: 'Batch Size',
      btn_start_train: 'Start Training',
      train_busy: 'Training…',
      btn_stop: 'Stop',
      train_stage_init: 'Init',
      train_stage_preprocess: 'Preprocess',
      train_stage_features: 'Features',
      train_stage_train: 'Train',
      train_stage_finalize: 'Finalize',
      train_loss_label: 'Training Loss',
      train_log_label: 'Log',
      train_done_title: 'Training Complete!',
      train_done_sub: 'Voice model saved to your profiles.',
      btn_goto_voice: 'Go to Voice Profiles →',
      // Modal
      modal_working: 'Working…',
      modal_discard: 'Discard',
      modal_apply: 'Apply',
    },
    kk: {
      nav_studio: 'Студия',
      nav_library: 'Кітапхана',
      nav_evaluate: 'Бағалау',
      nav_voice: 'Дауыс',
      nav_train: 'Дауыс үйрету',
      studio_title: 'Жасау',
      studio_sub: 'Музыкаңызды сипаттаңыз, сөздер қосыңыз, жасаңыз',
      label_description: 'Музыка сипаттамасы',
      btn_enhance: 'AI-мен жақсарту',
      btn_write_ai: 'AI-мен жазу',
      label_lyrics: 'Сөздер',
      label_optional: 'міндетті емес',
      label_optional_parens: '(міндетті емес)',
      label_style: 'Стиль',
      style_hint: 'қосу үшін басыңыз',
      theme_light: 'Жарық тақырып',
      theme_dark: 'Күңгірт тақырып',
      skin_label: 'Скин',
      skin_default: 'Негізгі',
      skin_amber: 'Амбер',
      skin_paper: 'Қағаз',
      label_parameters: 'Параметрлер',
      auto_duration: 'Авто',
      param_duration: 'Ұзақтық',
      param_guidance: 'Бағыттау',
      param_steps: 'Қадамдар',
      param_bpm: 'БПМ',
      param_key: 'Тональность',
      param_seed: 'Тұқым',
      btn_generate: 'Трек жасау',
      btn_cancel: 'Болдырмау',
      player_empty_title: 'Ештеңе ойнатылмайды',
      player_empty_sub: 'Жасалған трек осында пайда болады',
      btn_download: 'Жүктеу',
      btn_save: 'Сақтау',
      queue_title: 'Кезек',
      activity_title: 'Белсенділік',
      btn_clear: 'Тазалау',
      library_title: 'Кітапхана',
      eval_title: 'Бағалау',
      btn_run_comparison: 'Салыстыруды іске қосу',
      badge_speed: 'Жылдам',
      badge_recommended: 'Ұсынылған',
      badge_quality: 'Жоғары сапа',
      badge_new: 'ЖАҢА',
      unit_sec: 'сек',
      range_creative: 'шығармашыл',
      range_strict: 'қатаң',
      range_fast: 'жылдам',
      range_quality: 'сапалы',
      status_online: 'Желіде',
      status_offline: 'Желіден тыс',
      status_connecting: 'Қосылуда…',
      queue_idle: 'Белсенді генерация жоқ',
      queue_waiting: 'күтуде',
      log_connected: 'Қосылды',
      log_started: 'Генерация басталды',
      log_ready: 'Трек дайын',
      log_ready_init: 'Дайын.',
      log_saved: 'Кітапханаға сақталды',
      log_downloading: 'Жүктелуде...',
      log_prompt_applied: 'Сипаттама қолданылды',
      log_lyrics_applied: 'Сөздер қолданылды',
      stat_total: 'Барлығы жасалды',
      stat_avg_bpm: 'Орт. БПМ',
      stat_avg_dur: 'Орт. ұзақтық',
      stat_avg_energy: 'Орт. энергия',
      stat_last7: 'Соңғы 7 күн',
      stat_top_moods: 'Үздік көңіл-күйлер',
      lib_history: 'Тректер тарихы',
      lib_search_placeholder: 'Сипаттамаларды іздеу...',
      lib_empty: 'Тректер жоқ. Алдымен бірдеңе жасаңыз.',
      btn_refresh: 'Жаңарту',
      btn_load_more: 'Көбірек жүктеу',
      btn_export_csv: 'CSV экспорттау',
      eval_sub: 'Бір промптты әртүрлі сапа параметрлерімен салыстырыңыз',
      eval_prompt_label: 'Бағалауға арналған промпт',
      eval_presets_title: 'Салыстыру алдын ала орнатулары',
      eval_presets_hint: 'әр нұсқаны реттеңіз',
      eval_variant_a: 'А нұсқасы — Жылдам',
      eval_variant_b: 'В нұсқасы — Теңдестірілген',
      eval_variant_c: 'С нұсқасы — Сапалы',
      // Vocals selector
      label_vocals: 'Вокал',
      vocal_auto: 'Авто',
      vocal_female: 'Әйел',
      vocal_male: 'Ер',
      vocal_mixed: 'Аралас',
      // UI controls
      btn_history: 'Тарих',
      badge_idle: 'күту',
      badge_active: 'белсенді',
      btn_stems: 'Стемдер',
      btn_normalize: 'Нормалау',
      vs_hint: 'Ән бөлімдеріне әртүрлі дауыстар тағайындаңыз.',
      vs_processing: 'Өңделуде…',
      btn_apply_sections: 'Дауыс бөлімдерін қолдану',
      lib_tab_all: 'Барлығы',
      lib_tab_favs: 'Таңдаулылар',
      lib_tab_vs: 'Дауыс бөлімдері',
      lib_sort_label: 'Сұрыптау:',
      sort_newest: 'Жаңасы',
      sort_oldest: 'Ескісі',
      btn_select: 'Таңдау',
      btn_bulk_zip: 'ZIP экспорт',
      btn_bulk_del: 'Жою',
      metric_gen_time: 'Жасау уақыты',
      eval_generating: 'Жасалуда…',
      eval_ready: 'Дайын',
      eval_failed: 'Қате',
      // Voice toggle
      voice_toggle_label: 'Менің дауысыммен айту',
      // Video panel
      video_no_track: 'Трек жоқ',
      video_no_track_sub: 'Алдымен музыкалық трек жасаңыз, содан кейін бейне жасау үшін оралыңыз.',
      video_title: 'AI бейне генерациясы',
      video_sub: 'Wan2.1 әр лирикалық бөлім үшін кинематографиялық клип жасайды, содан кейін аудиомен біріктіреді.',
      video_generating: 'Бейне жасалуда',
      btn_gen_video: 'Бейне жасау',
      btn_dl_mp4: 'MP4 жүктеу',
      btn_play_audio: 'Аудиомен ойнату',
      btn_regenerate: 'Қайта жасау',
      // Analysis
      analyzing_audio: 'Аудио талдануда…',
      metric_bpm: 'БПМ',
      metric_key: 'Тональность',
      metric_mood: 'Көңіл-күй',
      metric_energy: 'Энергия',
      spectrum_bass: 'Бас',
      spectrum_mid: 'Орта',
      spectrum_treble: 'Жоғары',
      // Voice page
      voice_title: 'Менің дауысым',
      voice_sub: 'Дауыс профилін сақтаңыз — содан кейін Студияда кез келген әнді өз дауысыңызбен айтыңыз',
      voice_svc_sub: 'Нөлден · кез келген жазба',
      voice_rvc_label: 'RVC моделі',
      voice_rvc_sub: 'Үйретілген модельді жүктеу',
      voice_step_record: '10–60 секундтық дауыс үлгісін жазыңыз немесе жүктеңіз',
      btn_start_record: 'Жазуды бастау',
      btn_stop_record: 'Жазуды тоқтату',
      voice_or: 'немесе',
      btn_upload_audio: 'Аудио жүктеу',
      btn_save_profile: 'Профильді сақтау',
      voice_isolating: 'Вокалды бөлу',
      voice_denoising: 'Шуды азайту',
      voice_normalizing: 'Нормалау',
      voice_rvc_step: 'Үйретілген RVC модель файлын жүктеңіз',
      btn_upload_pth: '.pth модельді жүктеу',
      btn_upload_index: '.index жүктеу',
      btn_save_rvc: 'RVC модельді сақтау',
      voice_saved: 'Сақталған дауыстар',
      voice_empty: 'Дауыстар жоқ — жоғарыда жазыңыз немесе жүктеңіз.',
      // Train page
      train_title: 'Дауыс моделін үйрету',
      train_sub: 'Жеке спикер моделін үйрету үшін бірнеше дауыс жазбаларын жүктеңіз',
      train_data_label: 'Үйрету деректері',
      train_drop_title: 'Аудио файлдарды осында сүйреңіз немесе',
      train_drop_link: 'шолу',
      train_drop_hint: 'WAV · MP3 · FLAC · Бірнеше файл ұсынылады (әрқайсысы 5–60 сек)',
      train_params_label: 'Үйрету параметрлері',
      train_model_name: 'Модель атауы',
      train_epochs: 'Эпохалар',
      train_epochs_hint: 'Көп = сапасы жақсы, баяу',
      train_sample_rate: 'Іріктеу жиілігі',
      train_batch_size: 'Пакет өлшемі',
      btn_start_train: 'Үйретуді бастау',
      train_busy: 'Үйретілуде…',
      btn_stop: 'Тоқтату',
      train_stage_init: 'Баст.',
      train_stage_preprocess: 'Алдын ала',
      train_stage_features: 'Белгілер',
      train_stage_train: 'Үйрету',
      train_stage_finalize: 'Аяқтау',
      train_loss_label: 'Үйрету шығыны',
      train_log_label: 'Журнал',
      train_done_title: 'Үйрету аяқталды!',
      train_done_sub: 'Дауыс моделі профильдеріңізге сақталды.',
      btn_goto_voice: 'Дауыс профильдеріне өту →',
      // Modal
      modal_working: 'Жұмыс істеуде…',
      modal_discard: 'Жою',
      modal_apply: 'Қолдану',
    },
    ru: {
      nav_studio: 'Студия',
      nav_library: 'Библиотека',
      nav_evaluate: 'Оценка',
      nav_voice: 'Голос',
      nav_train: 'Обучение',
      studio_title: 'Создать',
      studio_sub: 'Опишите трек, добавьте текст, нажмите создать',
      label_description: 'Описание музыки',
      btn_enhance: 'Улучшить с AI',
      btn_write_ai: 'Написать с AI',
      label_lyrics: 'Текст',
      label_optional: 'необязательно',
      label_optional_parens: '(необязательно)',
      label_style: 'Стиль',
      style_hint: 'нажмите для добавления',
      theme_light: 'Светлая тема',
      theme_dark: 'Тёмная тема',
      skin_label: 'Скин',
      skin_default: 'Стандарт',
      skin_amber: 'Амбер',
      skin_paper: 'Бумага',
      label_parameters: 'Параметры',
      auto_duration: 'Авто',
      param_duration: 'Длительность',
      param_guidance: 'Управление',
      param_steps: 'Шаги',
      param_bpm: 'БПМ',
      param_key: 'Тональность',
      param_seed: 'Зерно',
      btn_generate: 'Создать трек',
      btn_cancel: 'Отмена',
      player_empty_title: 'Ничего не играет',
      player_empty_sub: 'Сгенерированный трек появится здесь',
      btn_download: 'Скачать',
      btn_save: 'Сохранить',
      queue_title: 'Очередь',
      activity_title: 'Активность',
      btn_clear: 'Очистить',
      library_title: 'Библиотека',
      eval_title: 'Оценка',
      btn_run_comparison: 'Запустить сравнение',
      badge_speed: 'Быстрый',
      badge_recommended: 'Рекомендуемый',
      badge_quality: 'Высокое качество',
      badge_new: 'НОВЫЙ',
      unit_sec: 'сек',
      range_creative: 'творческий',
      range_strict: 'строгий',
      range_fast: 'быстро',
      range_quality: 'качество',
      status_online: 'В сети',
      status_offline: 'Не в сети',
      status_connecting: 'Подключение…',
      queue_idle: 'Нет активной генерации',
      queue_waiting: 'ожидают',
      log_connected: 'Подключено',
      log_started: 'Генерация началась',
      log_ready: 'Трек готов',
      log_ready_init: 'Готово.',
      log_saved: 'Сохранено в библиотеку',
      log_downloading: 'Загрузка...',
      log_prompt_applied: 'Описание применено',
      log_lyrics_applied: 'Текст применён',
      stat_total: 'Всего создано',
      stat_avg_bpm: 'Средн. БПМ',
      stat_avg_dur: 'Средн. длит.',
      stat_avg_energy: 'Средн. энергия',
      stat_last7: 'За 7 дней',
      stat_top_moods: 'Топ настроений',
      lib_history: 'История треков',
      lib_search_placeholder: 'Поиск по описанию...',
      lib_empty: 'Треков пока нет. Сначала создайте что-нибудь.',
      btn_refresh: 'Обновить',
      btn_load_more: 'Загрузить ещё',
      btn_export_csv: 'Экспорт CSV',
      eval_sub: 'Сравните один промпт при разных настройках качества',
      eval_prompt_label: 'Промпт для оценки',
      eval_presets_title: 'Наборы для сравнения',
      eval_presets_hint: 'настройте каждый вариант',
      eval_variant_a: 'Вариант А — Быстрый',
      eval_variant_b: 'Вариант В — Сбалансированный',
      eval_variant_c: 'Вариант С — Качественный',
      // Vocals selector
      label_vocals: 'Вокал',
      vocal_auto: 'Авто',
      vocal_female: 'Женский',
      vocal_male: 'Мужской',
      vocal_mixed: 'Микс',
      // UI controls
      btn_history: 'История',
      badge_idle: 'ожид.',
      badge_active: 'активно',
      btn_stems: 'Стемы',
      btn_normalize: 'Нормализация',
      vs_hint: 'Назначьте разные голоса для каждой части песни.',
      vs_processing: 'Обработка…',
      btn_apply_sections: 'Применить голосовые секции',
      lib_tab_all: 'Все',
      lib_tab_favs: 'Избранное',
      lib_tab_vs: 'Голосовые секции',
      lib_sort_label: 'Сорт.:',
      sort_newest: 'Новые',
      sort_oldest: 'Старые',
      btn_select: 'Выбрать',
      btn_bulk_zip: 'Экспорт ZIP',
      btn_bulk_del: 'Удалить',
      metric_gen_time: 'Время ген.',
      eval_generating: 'Генерация…',
      eval_ready: 'Готово',
      eval_failed: 'Ошибка',
      // Voice toggle
      voice_toggle_label: 'Петь моим голосом',
      // Video panel
      video_no_track: 'Нет трека',
      video_no_track_sub: 'Сначала создайте музыкальный трек, затем вернитесь сюда для создания видео.',
      video_title: 'AI генерация видео',
      video_sub: 'Wan2.1 создаст кинематографический клип для каждой части текста и соединит их с вашим аудио.',
      video_generating: 'Создание видео',
      btn_gen_video: 'Создать видео',
      btn_dl_mp4: 'Скачать MP4',
      btn_play_audio: 'Воспроизвести с аудио',
      btn_regenerate: 'Пересоздать',
      // Analysis
      analyzing_audio: 'Анализ аудио…',
      metric_bpm: 'БПМ',
      metric_key: 'Тональность',
      metric_mood: 'Настроение',
      metric_energy: 'Энергия',
      spectrum_bass: 'Бас',
      spectrum_mid: 'Средние',
      spectrum_treble: 'Высокие',
      // Voice page
      voice_title: 'Мой голос',
      voice_sub: 'Сохраните голосовой профиль — затем используйте в Студии, чтобы петь любую песню своим голосом',
      voice_svc_sub: 'Без обучения · любая запись',
      voice_rvc_label: 'Модель RVC',
      voice_rvc_sub: 'Загрузить обученную модель',
      voice_step_record: 'Запишите или загрузите образец голоса 10–60 секунд',
      btn_start_record: 'Начать запись',
      btn_stop_record: 'Остановить запись',
      voice_or: 'или',
      btn_upload_audio: 'Загрузить аудио',
      btn_save_profile: 'Сохранить профиль',
      voice_isolating: 'Извлечение вокала',
      voice_denoising: 'Шумоподавление',
      voice_normalizing: 'Нормализация',
      voice_rvc_step: 'Загрузите файл обученной модели RVC',
      btn_upload_pth: 'Загрузить .pth модель',
      btn_upload_index: 'Загрузить .index',
      btn_save_rvc: 'Сохранить модель RVC',
      voice_saved: 'Сохранённые голоса',
      voice_empty: 'Голосов пока нет — запишите или загрузите выше.',
      // Train page
      train_title: 'Обучение голосовой модели',
      train_sub: 'Загрузите несколько голосовых записей для обучения модели говорящего',
      train_data_label: 'Данные для обучения',
      train_drop_title: 'Перетащите аудиофайлы сюда или',
      train_drop_link: 'выберите файлы',
      train_drop_hint: 'WAV · MP3 · FLAC · Рекомендуется несколько файлов (5–60 сек каждый)',
      train_params_label: 'Параметры обучения',
      train_model_name: 'Название модели',
      train_epochs: 'Эпохи',
      train_epochs_hint: 'Больше = лучше качество, медленнее',
      train_sample_rate: 'Частота дискретизации',
      train_batch_size: 'Размер пакета',
      btn_start_train: 'Начать обучение',
      train_busy: 'Обучение…',
      btn_stop: 'Стоп',
      train_stage_init: 'Иниц.',
      train_stage_preprocess: 'Подготовка',
      train_stage_features: 'Признаки',
      train_stage_train: 'Обучение',
      train_stage_finalize: 'Завершение',
      train_loss_label: 'Потери обучения',
      train_log_label: 'Журнал',
      train_done_title: 'Обучение завершено!',
      train_done_sub: 'Голосовая модель сохранена в ваши профили.',
      btn_goto_voice: 'Перейти к голосовым профилям →',
      // Modal
      modal_working: 'Обработка…',
      modal_discard: 'Отменить',
      modal_apply: 'Применить',
    },
  };

  S.uiLang = localStorage.getItem('maz_ui_lang') || 'en';

  function t(key) {
    return (TRANSLATIONS[S.uiLang] || TRANSLATIONS.en)[key] || key;
  }

  function applyTranslations() {
    // Apply all data-i18n elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.dataset.i18n;
      const val = t(key);
      if (val) el.textContent = val;
    });

    // Update placeholders
    const promptEl = $('prompt');
    if (promptEl) {
      const placeholders = {
        en: 'Describe the music… genre, mood, instruments, era, energy level…',
        kk: 'Музыканы сипаттаңыз… жанр, көңіл-күй, аспаптар, эра…',
        ru: 'Опишите музыку… жанр, настроение, инструменты, эпоха…',
      };
      promptEl.placeholder = placeholders[S.uiLang] || placeholders.en;
    }

    const lyricsEl = $('lyrics');
    if (lyricsEl) {
      const placeholders = {
        en: '[Verse 1]\nYour lyrics here…\n\n[Chorus]\nSection tags give best results',
        kk: '[1-шумақ]\nСөздеріңіз осында…\n\n[Қайырма]\nБөлім белгілері жақсы нәтиже береді',
        ru: '[Куплет 1]\nВаш текст здесь…\n\n[Припев]\nТеги разделов дают лучший результат',
      };
      lyricsEl.placeholder = placeholders[S.uiLang] || placeholders.en;
    }

    // Update status text if connected
    const statusLabel = $('status-text');
    if (statusLabel && statusLabel.textContent !== 'Reconnecting') {
      // Will be updated next health check
    }

    // Update lang switcher buttons
    $$('.lang-switch-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === S.uiLang);
    });

    // Update BPM/Key/Seed placeholders
    const bpmEl = $('bpm'), keyEl = $('key'), seedEl = $('seed');
    const autoText = S.uiLang === 'kk' ? 'авто' : S.uiLang === 'ru' ? 'авто' : 'auto';
    const randomText = S.uiLang === 'kk' ? 'кездейсоқ' : S.uiLang === 'ru' ? 'случайно' : 'random';
    if (bpmEl) bpmEl.placeholder = autoText;
    if (keyEl) keyEl.placeholder = autoText;
    if (seedEl) seedEl.placeholder = randomText;

    // Update eval placeholders (IDs: eval-prompt and eval-lyrics in HTML)
    const evalPromptTa = $('eval-prompt');
    if (evalPromptTa) evalPromptTa.placeholder =
      S.uiLang === 'kk' ? 'Бағалау үшін музыканы сипаттаңыз...'
        : S.uiLang === 'ru' ? 'Опишите музыку для оценки...'
          : 'Enter a music description to compare across different settings...';

    const evalLyricsTa = $('eval-lyrics');
    if (evalLyricsTa) evalLyricsTa.placeholder =
      S.uiLang === 'kk' ? 'Қосымша сөздер...'
        : S.uiLang === 'ru' ? 'Необязательный текст...'
          : 'Optional lyrics...';

    // Update search placeholder
    const libSearch = $('lib-search');
    if (libSearch) libSearch.placeholder = t('lib_search_placeholder');

    // Fix dice button to just show a simple symbol
    const diceBtn = $('btn-rand');
    if (diceBtn) diceBtn.textContent = '?';

    // Update page title
    document.title = S.uiLang === 'kk' ? 'MAZ — Музыка генераторы'
      : S.uiLang === 'ru' ? 'MAZ — Генератор музыки'
        : 'MAZ — Music Generator';

    // Voice page placeholders
    const voiceNameEl = $('voice-name-input');
    if (voiceNameEl) voiceNameEl.placeholder =
      S.uiLang === 'kk' ? 'Дауысқа ат беріңіз (мыс. Менің дауысым)'
        : S.uiLang === 'ru' ? 'Назовите голос (напр. Мой голос)'
          : 'Name this voice (e.g. My Voice)';

    const rvcNameEl = $('rvc-name-input');
    if (rvcNameEl) rvcNameEl.placeholder =
      S.uiLang === 'kk' ? 'Дауыс моделіне ат беріңіз'
        : S.uiLang === 'ru' ? 'Назовите голосовую модель'
          : 'Name this voice model';

    // Train page placeholders
    const trainNameEl = $('train-name');
    if (trainNameEl) trainNameEl.placeholder =
      S.uiLang === 'kk' ? 'Менің үйретілген дауысым'
        : S.uiLang === 'ru' ? 'Мой обученный голос'
          : 'My Trained Voice';

    // Voice select dropdown
    const voiceSel = $('voice-profile-select');
    if (voiceSel && voiceSel.options[0]) {
      voiceSel.options[0].textContent =
        S.uiLang === 'kk' ? 'Дауысты таңдаңыз…'
          : S.uiLang === 'ru' ? 'Выберите голос…'
            : 'Select voice…';
    }
  }

  // Sidebar language switcher — controls interface language only (S.uiLang)
  // Does NOT affect the lyrics language selector (S.lang) — those are independent
  $$('.lang-switch-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      S.uiLang = btn.dataset.lang;
      localStorage.setItem('maz_ui_lang', S.uiLang);
      applyTranslations();
      log(S.uiLang === 'kk' ? 'Тіл өзгертілді: Қазақша'
        : S.uiLang === 'ru' ? 'Язык изменён: Русский'
          : 'Language changed: English', 'success');
    });
  });

  // Apply on load
  applyTranslations();

  /* Init */
  connectWS();

  /* ── Voice Profiles ──────────────────────────────────────── */
  let voiceMediaRecorder   = null;
  let voiceChunks          = [];
  let voiceTimerInterval   = null;
  let voiceBlob            = null;
  let rvcModelFile         = null;
  let rvcIndexFile         = null;
  let filterStageTimeouts  = [];
  const MAX_RECORD_SECS    = 60;

  // ── SVC / RVC tab switching ──────────────────────────────────
  $$('.voice-type-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $$('.voice-type-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const type = tab.dataset.type;
      if (type === 'rvc') {
        $('voice-svc-panel').classList.add('hidden');
        $('voice-rvc-panel').classList.remove('hidden');
      } else {
        $('voice-rvc-panel').classList.add('hidden');
        $('voice-svc-panel').classList.remove('hidden');
      }
      $('voice-register-status').textContent = '';
    });
  });

  async function drawVoiceWaveform(blob) {
    const canvas = $('voice-waveform');
    if (!canvas) return;
    try {
      const W = canvas.parentElement.clientWidth || 480;
      canvas.width = W;
      canvas.height = 60;
      const arrayBuf = await blob.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded  = await audioCtx.decodeAudioData(arrayBuf);
      audioCtx.close();
      const data  = decoded.getChannelData(0);
      const barW  = 2, gap = 1;
      const bars  = Math.floor(W / (barW + gap));
      const step  = Math.max(1, Math.floor(data.length / bars));
      const mid   = 30;
      const ctx   = canvas.getContext('2d');
      ctx.clearRect(0, 0, W, 60);
      ctx.fillStyle = 'rgba(29,185,84,.75)';
      for (let i = 0; i < bars; i++) {
        let peak = 0;
        for (let j = 0; j < step; j++) {
          const v = Math.abs(data[i * step + j] || 0);
          if (v > peak) peak = v;
        }
        const h = Math.max(2, peak * 54);
        ctx.fillRect(i * (barW + gap), mid - h / 2, barW, h);
      }
      canvas.classList.remove('hidden');
    } catch (_) {
      canvas.classList.add('hidden');
    }
  }

  function startFilterProgress() {
    filterStageTimeouts.forEach(clearTimeout);
    filterStageTimeouts = [];
    [0, 1, 2].forEach(i => {
      const el = $(`vfs-${i}`);
      if (el) el.classList.remove('active', 'done');
    });
    const activate = i => { const el = $(`vfs-${i}`); if (el) el.classList.add('active'); };
    const complete = i => {
      const el = $(`vfs-${i}`);
      if (el) { el.classList.remove('active'); el.classList.add('done'); }
    };
    activate(0);
    filterStageTimeouts.push(setTimeout(() => { complete(0); activate(1); }, 22000));
    filterStageTimeouts.push(setTimeout(() => { complete(1); activate(2); }, 40000));
    $('voice-filter-progress').classList.remove('hidden');
    $('voice-register-status').textContent = '';
  }

  function stopFilterProgress(success) {
    filterStageTimeouts.forEach(clearTimeout);
    filterStageTimeouts = [];
    if (success) {
      [0, 1, 2].forEach(i => {
        const el = $(`vfs-${i}`);
        if (el) { el.classList.remove('active'); el.classList.add('done'); }
      });
      setTimeout(() => $('voice-filter-progress').classList.add('hidden'), 900);
    } else {
      [0, 1, 2].forEach(i => {
        const el = $(`vfs-${i}`);
        if (el) el.classList.remove('active', 'done');
      });
      $('voice-filter-progress').classList.add('hidden');
    }
  }

  // Load profiles list on Voice tab open + request mic permission
  $$('.sidebar-item').forEach(btn => {
    if (btn.dataset.view === 'voice') btn.addEventListener('click', () => {
      loadVoiceProfiles();
      requestMicPermission();
    });
  });

  async function requestMicPermission() {
    const statusEl = $('voice-register-status');

    // If already granted, just confirm quietly
    if (navigator.permissions) {
      try {
        const result = await navigator.permissions.query({ name: 'microphone' });
        if (result.state === 'granted') {
          if (statusEl) statusEl.textContent = '';
          return;
        }
        if (result.state === 'denied') {
          if (statusEl) statusEl.textContent = '⚠ Microphone access is blocked. Please allow it in your browser settings.';
          return;
        }
      } catch { /* permissions API not supported, fall through to getUserMedia */ }
    }

    // Prompt the browser permission dialog
    try {
      if (statusEl) statusEl.textContent = '🎙 Requesting microphone access…';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop()); // immediately release — we just needed the permission
      if (statusEl) statusEl.textContent = '✓ Microphone access granted.';
      setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 3000);
    } catch (err) {
      if (statusEl) statusEl.textContent = '⚠ Microphone access denied. Recording will not work.';
    }
  }

  // Also populate the Studio dropdown whenever Voice tab is opened or on page load
  async function loadVoiceProfiles() {
    try {
      const r = await fetch('/voice/profiles');
      const d = await r.json();
      _voiceProfilesCache = d.profiles || [];
      renderVoiceProfilesList(d.profiles);
      renderVoiceProfileSelect(d.profiles);
    } catch (e) { console.error('[voice] Failed to load profiles:', e); }
  }

  function renderVoiceProfilesList(profiles) {
    const list = $('voice-profiles-list');
    if (!list) return;
    if (!profiles.length) {
      list.innerHTML = `<p class="voice-empty-hint">${t('voice_empty')}</p>`;
      return;
    }
    list.innerHTML = profiles.map(p => {
      const isRvc = p.type === 'rvc';
      return `
      <div class="voice-profile-row" data-id="${escHtml(p.voice_id)}">
        <svg viewBox="0 0 24 24" fill="none" width="16" height="16" style="flex-shrink:0">
          <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" fill="currentColor" opacity=".7"/>
          <path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
        <span class="voice-profile-name">${escHtml(p.name)}</span>
        <span class="voice-badge ${isRvc ? 'voice-badge--rvc' : 'voice-badge--svc'}" style="font-size:10px;padding:2px 6px">${isRvc ? 'RVC' : 'SVC'}</span>
        <button class="voice-delete-btn" data-id="${escHtml(p.voice_id)}" title="Delete">✕</button>
      </div>`;
    }).join('');

    list.querySelectorAll('.voice-delete-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const id = btn.dataset.id;
        await fetch(`/voice/profiles/${id}`, { method: 'DELETE' });
        loadVoiceProfiles();
      });
    });
  }

  function renderVoiceProfileSelect(profiles) {
    const sel = $('voice-profile-select');
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = `<option value="">${S.uiLang === 'kk' ? 'Дауысты таңдаңыз…' : S.uiLang === 'ru' ? 'Выберите голос…' : 'Select voice…'}</option>` +
      profiles.map(p => {
        const tag = p.type === 'rvc' ? '[RVC] ' : '[SVC] ';
        return `<option value="${escHtml(p.voice_id)}">${tag}${escHtml(p.name)}</option>`;
      }).join('');
    if (prev) sel.value = prev;
  }

  // ── Recording ────────────────────────────────────────────────
  $('btn-record').addEventListener('click', async () => {
    if (voiceMediaRecorder && voiceMediaRecorder.state === 'recording') {
      voiceMediaRecorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      voiceChunks  = [];
      voiceMediaRecorder = new MediaRecorder(stream);

      voiceMediaRecorder.ondataavailable = e => { if (e.data.size > 0) voiceChunks.push(e.data); };
      voiceMediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        clearInterval(voiceTimerInterval);
        $('voice-timer').classList.add('hidden');
        $('btn-record-label').textContent = t('btn_start_record');
        $('btn-record').classList.remove('recording');

        voiceBlob = new Blob(voiceChunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(voiceBlob);
        const preview = $('voice-preview');
        preview.src = url;
        preview.classList.remove('hidden');
        $('voice-save-row').classList.remove('hidden');
        $('voice-register-status').textContent = '';
        drawVoiceWaveform(voiceBlob);
      };

      voiceMediaRecorder.start(100);
      $('btn-record-label').textContent = t('btn_stop_record');
      $('btn-record').classList.add('recording');
      $('voice-timer').classList.remove('hidden');
      $('voice-preview').classList.add('hidden');
      $('voice-save-row').classList.add('hidden');

      let elapsed = 0;
      voiceTimerInterval = setInterval(() => {
        elapsed++;
        const m = Math.floor(elapsed / 60);
        const s = String(elapsed % 60).padStart(2, '0');
        $('voice-timer-label').textContent = `${m}:${s} / 1:00`;
        if (elapsed >= MAX_RECORD_SECS) voiceMediaRecorder.stop();
      }, 1000);

    } catch (err) {
      $('voice-register-status').textContent = `⚠ Microphone access denied: ${err.message}`;
    }
  });

  // ── SVC file upload ──────────────────────────────────────────
  $('voice-file-input').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    voiceBlob = f;
    $('voice-file-label').innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" width="16" height="16">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"
          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      ${escHtml(f.name)}`;
    const url = URL.createObjectURL(f);
    const preview = $('voice-preview');
    preview.src = url;
    preview.classList.remove('hidden');
    $('voice-save-row').classList.remove('hidden');
    $('voice-register-status').textContent = '';
    drawVoiceWaveform(f);
  });

  // ── RVC file uploads ─────────────────────────────────────────
  $('rvc-model-input').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    rvcModelFile = f;
    $('rvc-model-label').innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" width="16" height="16">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"
          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      ${escHtml(f.name)}`;
    $('voice-rvc-save-row').classList.remove('hidden');
    $('voice-register-status').textContent = '';
  });

  $('rvc-index-input').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    rvcIndexFile = f;
    $('rvc-index-label').innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" width="16" height="16">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"
          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      ${escHtml(f.name)}`;
  });

  // ── Save SVC profile ─────────────────────────────────────────
  $('btn-save-voice').addEventListener('click', async () => {
    if (!voiceBlob) {
      $('voice-register-status').textContent = '⚠ Record or upload a voice first.';
      return;
    }
    const name = $('voice-name-input').value.trim() || 'My Voice';
    $('btn-save-voice').disabled = true;
    startFilterProgress();

    const form = new FormData();
    const filename = voiceBlob instanceof File ? voiceBlob.name : 'recording.webm';
    form.append('file', voiceBlob, filename);
    form.append('name', name);
    form.append('type', 'svc');

    try {
      const res = await fetch('/voice/register', { method: 'POST', body: form });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || res.statusText);
      }
      const data = await res.json();
      const cleanedNote = data.cleaned ? ' (vocals isolated)' : '';
      stopFilterProgress(true);
      $('voice-register-status').textContent = `✓ "${escHtml(data.name)}" saved${cleanedNote}.`;
      $('voice-save-row').classList.add('hidden');
      $('voice-name-input').value = '';
      voiceBlob = null;
      $('voice-preview').classList.add('hidden');
      $('voice-preview').src = '';
      $('voice-waveform').classList.add('hidden');
      loadVoiceProfiles();
    } catch (err) {
      stopFilterProgress(false);
      $('voice-register-status').textContent = `✗ ${err.message}`;
    } finally {
      $('btn-save-voice').disabled = false;
    }
  });

  // ── Save RVC model ────────────────────────────────────────────
  $('btn-save-rvc').addEventListener('click', async () => {
    if (!rvcModelFile) {
      $('voice-register-status').textContent = '⚠ Upload a .pth model file first.';
      return;
    }
    const name = $('rvc-name-input').value.trim() || 'RVC Voice';
    $('btn-save-rvc').disabled = true;
    $('voice-register-status').textContent = 'Saving RVC model…';

    const form = new FormData();
    form.append('file', rvcModelFile, rvcModelFile.name);
    form.append('name', name);
    form.append('type', 'rvc');
    if (rvcIndexFile) form.append('index_file', rvcIndexFile, rvcIndexFile.name);

    try {
      const res = await fetch('/voice/register', { method: 'POST', body: form });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || res.statusText);
      }
      const data = await res.json();
      $('voice-register-status').textContent = `✓ RVC model "${escHtml(data.name)}" saved.`;
      $('voice-rvc-save-row').classList.add('hidden');
      $('rvc-name-input').value = '';
      // Reset labels
      $('rvc-model-label').innerHTML = `<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg> Upload .pth Model <span class="rvc-required">*</span>`;
      $('rvc-index-label').innerHTML = `<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg> Upload .index <span class="rvc-optional">(optional)</span>`;
      $('rvc-model-input').value = '';
      $('rvc-index-input').value = '';
      rvcModelFile = null;
      rvcIndexFile = null;
      loadVoiceProfiles();
    } catch (err) {
      $('voice-register-status').textContent = `✗ ${err.message}`;
    } finally {
      $('btn-save-rvc').disabled = false;
    }
  });

  // ── Studio voice toggle ──────────────────────────────────────
  const voiceToggle = $('voice-toggle');
  const voiceSelect = $('voice-profile-select');

  voiceToggle.addEventListener('change', () => {
    if (voiceToggle.checked) {
      voiceSelect.classList.remove('hidden');
      loadVoiceProfiles();
    } else {
      voiceSelect.classList.add('hidden');
    }
  });

  // Load on page init in case profiles already exist
  loadVoiceProfiles();

  /* ══ TRAIN PAGE ══════════════════════════════════════════════ */
  const TR = {
    files:    [],
    jobId:    null,
    losses:   [],
    status:   null,   // 'running' | 'completed' | 'failed' | 'cancelled'
  };

  const TRAIN_STAGES = ['init', 'preprocess', 'extract_f0', 'extract_feats', 'train', 'finalize'];

  // Open Train page from sidebar
  $$('.sidebar-item').forEach(btn => {
    if (btn.dataset.view === 'train') {
      btn.addEventListener('click', () => {
        if (TR.jobId) pollTrainStatus();
      });
    }
  });

  // Redirect "Go to Voice Profiles" button
  $('btn-goto-voice')?.addEventListener('click', () => {
    $$('.sidebar-item').forEach(b => b.classList.remove('active'));
    $$('.page').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-view="voice"]').classList.add('active');
    $('page-voice').classList.add('active');
    loadVoiceProfiles();
  });

  // ── Drag-and-drop / file picker ──────────────────────────────
  const dropzone    = $('train-dropzone');
  const fileInput   = $('train-file-input');

  dropzone?.addEventListener('click', () => fileInput?.click());
  dropzone?.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('over'); });
  dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('over'));
  dropzone?.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('over');
    addTrainFiles([...e.dataTransfer.files]);
  });
  fileInput?.addEventListener('change', e => addTrainFiles([...e.target.files]));

  function addTrainFiles(incoming) {
    const audioTypes = /^audio\//;
    incoming.forEach(f => {
      if (audioTypes.test(f.type) || /\.(wav|mp3|flac|ogg|m4a|aac)$/i.test(f.name)) {
        if (!TR.files.find(x => x.name === f.name && x.size === f.size)) {
          TR.files.push(f);
        }
      }
    });
    renderTrainFileList();
  }

  function renderTrainFileList() {
    const list  = $('train-file-list');
    const count = $('train-file-count');
    if (!list) return;
    list.innerHTML = TR.files.map((f, i) => `
      <div class="train-file-item">
        <svg viewBox="0 0 20 20" fill="none" width="14" height="14" style="flex-shrink:0;opacity:.5">
          <path d="M13 2H6a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7l-3-5z" stroke="currentColor" stroke-width="1.5"/>
          <path d="M13 2v5h5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <span class="train-file-name">${escHtml(f.name)}</span>
        <span class="train-file-size">${(f.size / 1024).toFixed(0)} KB</span>
        <button class="train-file-rm" data-idx="${i}" title="Remove">✕</button>
      </div>`).join('');

    list.querySelectorAll('.train-file-rm').forEach(btn => {
      btn.addEventListener('click', () => {
        TR.files.splice(parseInt(btn.dataset.idx), 1);
        renderTrainFileList();
      });
    });

    if (count) count.textContent = TR.files.length ? `${TR.files.length} file${TR.files.length > 1 ? 's' : ''} selected` : '';
    if (fileInput) fileInput.value = '';
  }

  // ── Stage indicator helpers ───────────────────────────────────
  function setTrainStage(active) {
    let passed = true;
    TRAIN_STAGES.forEach(s => {
      const el = $(`tstage-${s}`);
      if (!el) return;
      if (s === active) {
        el.classList.add('active');
        el.classList.remove('done');
        passed = false;
      } else if (passed) {
        el.classList.add('done');
        el.classList.remove('active');
      } else {
        el.classList.remove('active', 'done');
      }
    });
  }

  function setAllStagesDone() {
    TRAIN_STAGES.forEach(s => {
      const el = $(`tstage-${s}`);
      if (el) { el.classList.add('done'); el.classList.remove('active'); }
    });
  }

  // ── Loss chart ────────────────────────────────────────────────
  function drawLossChart(losses) {
    const canvas = $('train-loss-canvas');
    if (!canvas || !losses.length) return;
    const dpr = window.devicePixelRatio || 1;
    const W   = canvas.parentElement.clientWidth || 600;
    const H   = 120;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const min = Math.min(...losses);
    const max = Math.max(...losses) || 1;
    const pad = { t: 8, b: 8, l: 4, r: 4 };
    const gW  = W - pad.l - pad.r;
    const gH  = H - pad.t - pad.b;

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (i / 4) * gH;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + gW, y); ctx.stroke();
    }

    // Line + fill
    ctx.beginPath();
    losses.forEach((v, i) => {
      const x = pad.l + (i / Math.max(losses.length - 1, 1)) * gW;
      const y = pad.t + gH - ((v - min) / (max - min || 1)) * gH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#1db954';
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    // Gradient fill below line
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + gH);
    grad.addColorStop(0, 'rgba(29,185,84,0.18)');
    grad.addColorStop(1, 'rgba(29,185,84,0)');
    ctx.lineTo(pad.l + gW, pad.t + gH);
    ctx.lineTo(pad.l, pad.t + gH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // ── Training log append ───────────────────────────────────────
  function appendTrainLog(msg) {
    const log = $('train-log');
    if (!log) return;
    const line = document.createElement('div');
    const ts   = new Date().toLocaleTimeString('en', { hour12: false });
    line.textContent = `[${ts}] ${msg}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    while (log.children.length > 120) log.removeChild(log.firstChild);
  }

  // ── WebSocket handler for training_progress ───────────────────
  function handleTrainingProgress(msg) {
    if (msg.job_id !== TR.jobId) return;
    const { stage, progress, losses, status } = msg;
    const msgText = msg.msg || '';

    // Update progress bar
    const pct = Math.round((progress || 0) * 100);
    const fill = $('train-progress-fill');
    const pctEl = $('train-progress-pct');
    const statusEl = $('train-status-msg');
    if (fill) fill.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';
    if (statusEl) statusEl.textContent = msgText;

    // Stage indicators
    if (stage && stage !== 'done' && stage !== 'error') setTrainStage(stage);

    // Log
    if (msgText) appendTrainLog(`[${stage}] ${msgText}`);

    // Loss chart
    if (losses && losses.length) {
      TR.losses = losses;
      $('train-chart-wrap')?.classList.remove('hidden');
      drawLossChart(losses);
    }

    // Completion
    if (status === 'completed' || stage === 'done') {
      TR.status = 'completed';
      setAllStagesDone();
      if (fill) fill.style.width = '100%';
      if (pctEl) pctEl.textContent = '100%';
      showTrainDone(msg.msg);
      resetTrainControls();
    } else if (status === 'failed' || stage === 'error') {
      TR.status = 'failed';
      if (statusEl) statusEl.textContent = '✗ ' + msgText;
      statusEl?.classList.add('train-status-error');
      appendTrainLog('✗ Training failed: ' + msgText);
      resetTrainControls();
    } else if (status === 'cancelled') {
      TR.status = 'cancelled';
      resetTrainControls();
    }
  }

  function showTrainDone(msg) {
    $('train-done-card')?.classList.remove('hidden');
    const sub = $('train-done-sub');
    if (sub) sub.textContent = msg || 'Voice model saved to your profiles.';
    loadVoiceProfiles();
  }

  function resetTrainControls() {
    $('btn-start-train').disabled = false;
    $('train-idle-label')?.classList.remove('hidden');
    $('train-busy-label')?.classList.add('hidden');
    $('btn-stop-train')?.classList.add('hidden');
  }

  // ── Poll training status on page open ─────────────────────────
  async function pollTrainStatus() {
    if (!TR.jobId) return;
    try {
      const r = await fetch(`/voice/train/${TR.jobId}`);
      if (!r.ok) return;
      const job = await r.json();
      const fill = $('train-progress-fill');
      const pctEl = $('train-progress-pct');
      const pct = Math.round((job.progress || 0) * 100);
      if (fill) fill.style.width = pct + '%';
      if (pctEl) pctEl.textContent = pct + '%';
      if (job.losses?.length) { TR.losses = job.losses; drawLossChart(job.losses); }
      if (job.stage && job.stage !== 'done' && job.stage !== 'error') setTrainStage(job.stage);
      if (job.status === 'completed') showTrainDone(job.msg);
    } catch { }
  }

  // ── Start training ────────────────────────────────────────────
  $('btn-start-train')?.addEventListener('click', async () => {
    if (!TR.files.length) {
      alert('Add at least one audio file first.');
      return;
    }
    const name    = ($('train-name').value.trim()) || 'My Trained Voice';
    const epochs  = parseInt($('train-epochs').value) || 80;
    const sr      = parseInt($('train-sr').value)     || 40000;
    const batch   = parseInt($('train-batch').value)  || 16;

    const form = new FormData();
    TR.files.forEach(f => form.append('files', f, f.name));
    form.append('name',        name);
    form.append('epochs',      String(epochs));
    form.append('sample_rate', String(sr));
    form.append('batch_size',  String(batch));

    $('btn-start-train').disabled = true;
    $('train-idle-label')?.classList.add('hidden');
    $('train-busy-label')?.classList.remove('hidden');
    $('btn-stop-train')?.classList.remove('hidden');
    $('train-progress-card')?.classList.remove('hidden');
    $('train-done-card')?.classList.add('hidden');
    $('train-log').innerHTML = '';
    TR.losses = [];
    TR.status = 'running';
    $('train-chart-wrap')?.classList.add('hidden');

    // Reset stages
    TRAIN_STAGES.forEach(s => {
      const el = $(`tstage-${s}`);
      if (el) el.classList.remove('active', 'done');
    });
    setTrainStage('init');

    appendTrainLog(`Starting training — ${TR.files.length} file(s), ${epochs} epochs`);

    try {
      const res = await fetch('/voice/train', { method: 'POST', body: form });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || res.statusText);
      }
      const data = await res.json();
      TR.jobId  = data.job_id;
      TR.status = 'running';
      log(`Voice training started — job ${data.job_id}`, 'ai');
      appendTrainLog(`Job ID: ${data.job_id}`);
    } catch (err) {
      $('train-status-msg').textContent = '✗ ' + err.message;
      appendTrainLog('✗ ' + err.message);
      resetTrainControls();
    }
  });

  // ── Stop training ─────────────────────────────────────────────
  $('btn-stop-train')?.addEventListener('click', async () => {
    if (!TR.jobId) return;
    try {
      await fetch(`/voice/train/${TR.jobId}`, { method: 'DELETE' });
      appendTrainLog('Training cancelled by user.');
      log('Voice training cancelled', 'warn');
    } catch { }
    resetTrainControls();
  });

}); // end DOMContentLoaded