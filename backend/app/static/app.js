let jobId = null;
let videos = [];
let currentVideoId = null;
let image = new Image();
let rois = {}; // { video_id: [[x,y],...] }
let mode = 'rect';
let isDrawing = false;
let startPt = null;
let polyPts = [];

const $ = (id) => document.getElementById(id);
const canvas = $('roiCanvas');
const ctx = canvas.getContext('2d');

function setStatus(text){ $('jobStatus').textContent = text; }
function setHint(id, text){ $(id).textContent = text || ''; }
function setProgress(v){
  const value = Math.max(0, Math.min(100, Number(v) || 0));
  $('progressBar').style.width = `${value}%`;
  $('progressPercent').textContent = `${Math.round(value)}%`;
}
function formatDuration(seconds){
  if(seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return '—';
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0
    ? `${String(hours).padStart(2,'0')}:${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')}`
    : `${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
}
function updateProcessDetails(data){
  $('processStage').textContent = data.stage || 'Обработка';
  $('processDevice').textContent = data.device || 'Определяется...';
  $('processElapsed').textContent = formatDuration(data.elapsed_seconds);
  $('processEta').textContent = data.status === 'done' ? '00:00' : formatDuration(data.eta_seconds);
}

function pointFromEvent(e){
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return [Math.round((e.clientX - rect.left) * sx), Math.round((e.clientY - rect.top) * sy)];
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(image.src) ctx.drawImage(image,0,0,canvas.width,canvas.height);
  const pts = rois[currentVideoId] || polyPts;
  if(pts && pts.length){
    ctx.save();
    ctx.lineWidth = Math.max(3, canvas.width/320);
    ctx.strokeStyle = '#ff1f1f';
    ctx.fillStyle = 'rgba(16,185,129,.22)';
    ctx.beginPath();
    pts.forEach((p,i)=> i ? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1]));
    if(pts.length > 2) ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fef08a'; ctx.strokeStyle = '#111827'; ctx.lineWidth = 2;
    pts.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p[0],p[1],7,0,Math.PI*2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = `${Math.max(14, canvas.width/70)}px Arial`;
      ctx.fillText(`P${i+1} (${p[0]},${p[1]})`, p[0]+10, p[1]+18);
      ctx.fillStyle = '#fef08a';
    });
    ctx.restore();
  }
  $('roiCoords').textContent = JSON.stringify(rois[currentVideoId] || polyPts || [], null, 2);
}

function rectPolygon(a,b){
  const x1 = Math.min(a[0], b[0]), x2 = Math.max(a[0], b[0]);
  const y1 = Math.min(a[1], b[1]), y2 = Math.max(a[1], b[1]);
  return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]];
}

canvas.addEventListener('mousedown', e => {
  if(!currentVideoId || !image.src) return;
  const p = pointFromEvent(e);
  if(mode === 'rect'){
    isDrawing = true; startPt = p; rois[currentVideoId] = rectPolygon(p,p); draw();
  } else {
    polyPts.push(p); rois[currentVideoId] = [...polyPts]; draw();
  }
});
canvas.addEventListener('mousemove', e => {
  if(mode !== 'rect' || !isDrawing) return;
  rois[currentVideoId] = rectPolygon(startPt, pointFromEvent(e)); draw();
});
window.addEventListener('mouseup', () => { isDrawing = false; });
canvas.addEventListener('dblclick', () => { if(mode === 'poly' && polyPts.length >= 3){ rois[currentVideoId] = [...polyPts]; polyPts = []; draw(); } });

$('rectModeBtn').onclick = () => { mode='rect'; polyPts=[]; $('roiModeLabel').textContent='режим: рамка'; };
$('polyModeBtn').onclick = () => { mode='poly'; polyPts=[]; $('roiModeLabel').textContent='режим: полигон, двойной клик завершает'; };
$('clearRoiBtn').onclick = () => { if(currentVideoId){ delete rois[currentVideoId]; polyPts=[]; draw(); } };
$('applyAllBtn').onclick = () => {
  const roi = rois[currentVideoId];
  if(!roi || roi.length < 3) return alert('Сначала нарисуйте ROI');
  videos.forEach(v => rois[v.video_id] = JSON.parse(JSON.stringify(roi)));
  draw();
  setHint('uploadHint', 'ROI применена ко всем видео.');
};

async function loadFrame(vid){
  currentVideoId = vid;
  const emptyCanvas = $('emptyCanvas');
  emptyCanvas.style.display = 'flex';
  emptyCanvas.textContent = 'Загружаю первый кадр...';
  setHint('uploadHint', 'Получаю первый кадр видео...');

  try {
    const url = `/api/jobs/${jobId}/videos/${encodeURIComponent(vid)}/frame?frame=0&t=${Date.now()}`;
    const response = await fetch(url, {cache: 'no-store'});
    if(!response.ok){
      const message = await response.text();
      throw new Error(message || `HTTP ${response.status}`);
    }

    const blob = await response.blob();
    if(!blob.type.startsWith('image/')){
      throw new Error(`Сервер вернул неподдерживаемый тип: ${blob.type || 'unknown'}`);
    }

    const objectUrl = URL.createObjectURL(blob);
    const nextImage = new Image();
    nextImage.onload = () => {
      if(image && image.dataset && image.dataset.objectUrl){
        URL.revokeObjectURL(image.dataset.objectUrl);
      }
      nextImage.dataset.objectUrl = objectUrl;
      image = nextImage;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      emptyCanvas.style.display = 'none';
      setHint('uploadHint', `Загружено видео: ${videos.length}. Выделите ROI полки.`);
      draw();
    };
    nextImage.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      emptyCanvas.textContent = 'Не удалось отобразить первый кадр';
      setHint('uploadHint', 'Кадр получен, но браузер не смог его декодировать.');
    };
    nextImage.src = objectUrl;
  } catch(error) {
    canvas.width = 0;
    canvas.height = 0;
    emptyCanvas.style.display = 'flex';
    emptyCanvas.textContent = 'Не удалось загрузить первый кадр';
    setStatus('Ошибка кадра');
    setHint('uploadHint', `Видео загружено, но первый кадр недоступен: ${error.message}`);
  }
}

$('videoSelect').onchange = (e) => loadFrame(e.target.value);

const dropzone = $('dropzone');
['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', e => { $('fileInput').files = e.dataTransfer.files; setHint('uploadHint', `${e.dataTransfer.files.length} файл(ов) выбрано.`); });
$('fileInput').addEventListener('change', e => setHint('uploadHint', `${e.target.files.length} файл(ов) выбрано.`));

$('uploadBtn').onclick = async () => {
  const files = $('fileInput').files;
  if(!files.length) return alert('Выберите файлы');
  const fd = new FormData();
  Array.from(files).forEach(f => fd.append('files', f));
  setStatus('Загрузка'); setHint('uploadHint','Загружаю файлы...');
  const res = await fetch('/api/jobs', {method:'POST', body:fd});
  if(!res.ok){ setStatus('Ошибка'); return setHint('uploadHint', await res.text()); }
  const data = await res.json();
  jobId = data.job_id; videos = data.videos;
  setStatus(`Job ${jobId}`);
  $('videoSelect').innerHTML = videos.map(v => `<option value="${v.video_id}">${v.video_id} · ${v.filename}</option>`).join('');
  if(videos.length){
    await loadFrame(videos[0].video_id);
  }
};

function processParams(){
  return {
    event_min_extension: parseFloat($('pExtension').value),
    event_min_action_signal: parseFloat($('pSignal').value),
    projection_alpha: parseFloat($('pProjection').value),
    heatmap_radius_px: parseInt($('pRadius').value),
    zone_percentile_thr: parseInt($('pPercentile').value),
    render_video: $('pRender').value === 'true'
  };
}

$('processBtn').onclick = async () => {
  if(!jobId) return alert('Сначала загрузите видео');
  const anyRoi = Object.keys(rois).length > 0;
  if(!anyRoi) return alert('Нарисуйте ROI полки');

  let payloadRois = {...rois};
  if(currentVideoId && rois[currentVideoId]) payloadRois['__global__'] = rois[currentVideoId];

  $('processBtn').disabled = true;
  setProgress(1); setStatus('Обработка'); setHint('processHint','Подготовка моделей и входных данных...');
  updateProcessDetails({stage:'Инициализация', device:'Определяется...', elapsed_seconds:0, eta_seconds:null});
  const res = await fetch(`/api/jobs/${jobId}/process`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({rois: payloadRois, params: processParams()})
  });
  if(!res.ok){ $('processBtn').disabled = false; setStatus('Ошибка'); return setHint('processHint', await res.text()); }
  pollStatus();
};

async function pollStatus(){
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/status?t=${Date.now()}`, {cache:'no-store'});
      if(!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStatus(data.status);
      setProgress(data.progress || 0);
      updateProcessDetails(data);
      setHint('processHint', data.message + (data.error ? `\n${data.error}` : ''));
      if(data.status === 'done'){
        clearInterval(timer); $('processBtn').disabled = false; setProgress(100); renderReport(data.report);
      }
      if(data.status === 'error'){
        clearInterval(timer); $('processBtn').disabled = false;
      }
    } catch(error) {
      clearInterval(timer);
      $('processBtn').disabled = false;
      setStatus('Ошибка статуса');
      setHint('processHint', `Не удалось получить прогресс: ${error.message}`);
    }
  }, 1000);
}

function renderReport(report){
  $('reportPanel').classList.add('show');
  $('downloadLink').href = `/api/jobs/${jobId}/download`;
  const overview = report.overview || [];
  $('reportSummary').innerHTML = overview.map(r => `<div class="summary-tile"><b>${r.value}</b><span>${r.metric}</span></div>`).join('');

  const links = [];
  if(report.html_report_url) links.push(`<a class="secondary link-btn" target="_blank" href="${report.html_report_url}">Открыть HTML-отчет</a>`);
  if(report.excel_report_url) links.push(`<a class="secondary link-btn" href="${report.excel_report_url}">Скачать Excel</a>`);
  if(report.all_zones_csv_url) links.push(`<a class="secondary link-btn" href="${report.all_zones_csv_url}">CSV зоны</a>`);
  if(report.all_events_csv_url) links.push(`<a class="secondary link-btn" href="${report.all_events_csv_url}">CSV события</a>`);
  $('reportLinks').innerHTML = links.join('');

  const videos = report.video_summary || [];
  $('videoCards').innerHTML = videos.map(v => {
    const img = v.summary_image ? `/outputs/${jobId}/${v.summary_image}` : '';
    return `<article class="video-card">
      ${img ? `<img src="${img}" alt="${v.video_id}">` : ''}
      <div>
        <h3>${v.video_id}</h3>
        <p><b>События:</b> ${v.n_events} · <b>Зоны:</b> ${v.n_zones}</p>
        <p><b>Главная зона:</b> ${v.top_zone || '-'} · ${v.top_action || '-'}</p>
        <p><b>Интерес:</b> ${Number(v.top_interest_0_100 || 0).toFixed(1)}</p>
        ${v.error ? `<p class="error">${v.error}</p>` : ''}
      </div>
    </article>`;
  }).join('');
  $('reportPanel').scrollIntoView({behavior:'smooth'});
}
