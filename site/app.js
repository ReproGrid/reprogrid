const languageSwitch = document.querySelector('#language-switch');

if (languageSwitch) {
  languageSwitch.addEventListener('click', () => {
    const target = new URL(languageSwitch.href);
    target.search = window.location.search;
    target.hash = window.location.hash;
    languageSwitch.href = target;
  });
}

const search = document.querySelector('#search');

if (search) {
  const list = document.querySelector('#paper-list');
  const cards = [...list.children];
  const buttons = [...document.querySelectorAll('[data-domain-button]')];
  const status = document.querySelector('#status-filter');
  const sort = document.querySelector('#sort');
  const count = document.querySelector('#match-count strong');
  const empty = document.querySelector('#empty-state');
  const pagination = document.querySelector('#pagination');
  const previous = document.querySelector('#previous-page');
  const next = document.querySelector('#next-page');
  const pageLabel = document.querySelector('#page-label');
  const pageSize = 12;
  const params = new URLSearchParams(window.location.search);
  let domain = buttons.some((button) => button.dataset.domainButton === params.get('domain')) ? params.get('domain') : 'all';
  let page = Math.max(1, Number.parseInt(params.get('page'), 10) || 1);
  search.value = params.get('q') ?? '';
  status.value = params.get('status') ?? 'all';
  sort.value = params.get('sort') ?? 'curated';
  if (!status.value) status.value = 'all';
  if (!sort.value) sort.value = 'curated';

  function render() {
    const query = search.value.trim();
    const terms = query.toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const matches = cards.filter((card) =>
      (domain === 'all' || card.dataset.domain === domain) &&
      (status.value === 'all' || card.dataset.status === status.value) &&
      terms.every((term) => card.dataset.keywords.toLocaleLowerCase().includes(term))
    );
    if (sort.value === 'newest') matches.sort((a, b) => b.dataset.date.localeCompare(a.dataset.date));
    if (sort.value === 'citations') matches.sort((a, b) => Number(b.dataset.citations) - Number(a.dataset.citations));
    const pages = Math.max(1, Math.ceil(matches.length / pageSize));
    page = Math.min(page, pages);
    for (const card of cards) card.hidden = true;
    for (const card of matches.slice((page - 1) * pageSize, page * pageSize)) {
      card.hidden = false;
      list.append(card);
    }
    for (const button of buttons) button.setAttribute('aria-pressed', String(button.dataset.domainButton === domain));
    count.textContent = matches.length;
    empty.hidden = matches.length > 0;
    pagination.hidden = pages === 1;
    previous.disabled = page === 1;
    next.disabled = page === pages;
    pageLabel.textContent = `${page} / ${pages}`;

    const url = new URL(window.location.href);
    for (const [key, value, defaultValue] of [
      ['q', query, ''], ['domain', domain, 'all'], ['status', status.value, 'all'],
      ['sort', sort.value, 'curated'], ['page', page, 1],
    ]) {
      if (value === defaultValue) url.searchParams.delete(key);
      else url.searchParams.set(key, value);
    }
    window.history.replaceState(null, '', url);
  }

  function filter() {
    page = 1;
    render();
  }

  search.addEventListener('input', filter);
  status.addEventListener('change', filter);
  sort.addEventListener('change', filter);
  for (const button of buttons) {
    button.addEventListener('click', () => {
      domain = button.dataset.domainButton;
      filter();
    });
  }
  document.querySelector('#reset-filters').addEventListener('click', () => {
    domain = 'all';
    search.value = '';
    status.value = 'all';
    sort.value = 'curated';
    filter();
    search.focus();
  });
  for (const [button, direction] of [[previous, -1], [next, 1]]) {
    button.addEventListener('click', () => {
      page += direction;
      render();
      list.scrollIntoView({ block: 'start' });
    });
  }
  render();
}

function drawTrace(figure, data) {
  const canvas = figure.querySelector('canvas');
  const context = canvas.getContext('2d');
  const { width, height } = canvas.getBoundingClientRect();
  if (width === 0 || height === 0 || !context) return;
  const scale = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  context.scale(scale, scale);

  const hero = figure.classList.contains('hero-trace');
  const left = 32;
  const right = width - 13;
  const top = 10;
  const bottom = height - 23;
  const maxTime = hero ? 24 : 80;
  const maxPower = 1.12;
  const x = (value) => left + value / maxTime * (right - left);
  const y = (value) => bottom - value / maxPower * (bottom - top);
  const ink = hero ? '#b9ccbf' : '#455b50';
  const grid = hero ? '#345246' : '#d8e4da';

  function line(x1, y1, x2, y2, color) {
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
  }

  context.font = '10px ui-monospace, monospace';
  context.fillStyle = ink;
  context.textAlign = 'right';
  for (const power of [0, 0.25, 0.5, 0.75, 1]) {
    line(left, y(power), right, y(power), grid);
    context.fillText(String(power), left - 5, y(power) + 3);
  }
  context.textAlign = 'center';
  for (let step = 0; step <= 4; step += 1) {
    const time = maxTime * step / 4;
    line(x(time), top, x(time), bottom, grid);
    context.fillText(String(time) + (step === 4 ? ' ms' : ''), x(time) - (step === 4 ? 8 : 0), bottom + 14);
  }
  const points = data[figure.dataset.case].filter(([time]) => time <= maxTime);
  for (const [column, color, dash] of [[1, hero ? '#a9bfbb' : '#72847f', [5, 4]], [2, hero ? '#d8ee95' : '#246a51', []]]) {
    context.beginPath();
    context.lineWidth = column === 2 ? 2.4 : 1.7;
    context.strokeStyle = color;
    context.setLineDash(dash);
    points.forEach(([time, ...values], index) => {
      const point = [x(time), y(values[column - 1])];
      if (index === 0) context.moveTo(...point);
      else context.lineTo(...point);
    });
    context.stroke();
  }
}

for (const figure of document.querySelectorAll('[data-trace]')) {
  fetch(figure.dataset.trace).then((response) => {
    if (!response.ok) throw new Error('Trace request failed');
    return response.json();
  }).then(({ cases }) => {
    const draw = () => drawTrace(figure, cases);
    new ResizeObserver(draw).observe(figure);
    draw();
    const buttons = [...(figure.closest('section')?.querySelectorAll('[data-trace-case]') ?? [])];
    for (const button of buttons) {
      button.addEventListener('click', () => {
        figure.dataset.case = button.dataset.traceCase;
        figure.querySelector('.trace-case').textContent = `L = ${Number(button.dataset.inductance).toFixed(1)} p.u.`;
        for (const item of buttons) item.setAttribute('aria-pressed', String(item === button));
        draw();
      });
    }
  }).catch(() => {
    figure.querySelector('canvas').hidden = true;
    const caption = figure.querySelector('figcaption');
    caption.append(' · ' + caption.dataset.error);
  });
}
