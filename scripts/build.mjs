import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const dist = path.join(root, 'dist');
const json = (file) => JSON.parse(readFileSync(path.join(root, file), 'utf8'));
const queue = json('research/candidate_queue.json');
const catalog = json('research/catalog.json');
const localized = json('research/localization.json');
const englishLabels = json('scripts/ui.en.json');
const spec = json('platform_spec.json');
const statuses = { pilot: ['试点有结果', 'pilot'], parked: ['暂缓', 'parked'], screened: ['候选待评估', 'screened'] };
const emptyVideos = [
  { title: '算法讲解', description: '待制作' },
  { title: '结果解读', description: '待制作' },
];
const seenDoi = new Set();
const seenSlug = new Set();
const papers = queue.papers.map((source) => {
  const entry = catalog.papers[source.doi];
  const domain = Object.keys(catalog.domains).find((id) => catalog.domains[id].keys.includes(source.domain));
  if (!entry || !domain || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(entry.slug)) {
    throw new Error(`Missing catalog entry or category for ${source.doi}`);
  }
  if (seenDoi.has(source.doi) || seenSlug.has(entry.slug)) throw new Error(`Duplicate DOI or slug: ${source.doi}`);
  seenDoi.add(source.doi);
  seenSlug.add(entry.slug);
  const translation = localized.papers[entry.slug];
  if (!translation || !localized.domains[domain]) throw new Error(`Missing English text for ${entry.slug}`);
  const experiment = entry.experiment;
  const paper = { source, translation, ...entry, domain, status: experiment ? 'pilot' : source.state, videos: entry.videos ?? emptyVideos };
  if (experiment) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(experiment)) throw new Error('Invalid experiment folder');
    paper.folder = `papers/${experiment}`;
    paper.report = json(`${paper.folder}/report.json`);
    paper.reportEn = json(`${paper.folder}/report.en.json`);
    if (paper.reportEn.videos.length !== paper.report.videos.length) throw new Error('Mismatched video translations');
    paper.results = json(`${paper.folder}/results/summary.json`);
    paper.videos = paper.report.videos;
    if (paper.results.doi !== source.doi) throw new Error('Experiment DOI mismatch');
  }
  if (!statuses[paper.status] || paper.videos.length < 2 || paper.videos.length > 3) {
    throw new Error(`Invalid status or video count for ${source.doi}`);
  }
  return paper;
});
if (Object.keys(catalog.papers).length !== seenDoi.size) throw new Error('Catalog and candidate queue differ');
papers.sort((a, b) => Number(Boolean(b.experiment)) - Number(Boolean(a.experiment)));

function escape(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}

function external(url, label) {
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' || parsed.username || parsed.password) throw new Error('Expected public HTTPS link');
  return `<a class="external-link" href="${escape(parsed.href)}" target="_blank" rel="noopener noreferrer">${escape(label)} <span aria-hidden="true">↗</span></a>`;
}

function renderLanguage(lang) {
  const english = lang === 'en';
  const siteBase = english ? '../' : './';
  const t = (text) => {
    if (!english) return text;
    if (!englishLabels[text]) throw new Error(`Missing English UI string: ${text}`);
    return englishLabels[text];
  };
  const domainTitle = (id) => english ? localized.domains[id] : catalog.domains[id].title;
  const entries = papers.map((paper) => {
    if (!english) return paper;
    const { description, tags, gap } = paper.translation;
    const report = paper.reportEn;
    const videos = report?.videos ?? paper.videos.map((video) =>
      emptyVideos.includes(video) ? { title: t(video.title), description: t(video.description) } : video
    );
    return { ...paper, originalTitle: paper.title, title: paper.source.title, description, tags, gap, report, videos };
  });

  function brand(base) {
    return `<a class="brand" href="${base}"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span>ReproGrid</span></a>`;
  }

  function page({ base, route = '', title, description, current, content }) {
    const assets = english ? base + '../' : base;
    const translationHref = base + (english ? '../' : 'en/') + route;
    const alternateLang = english ? 'zh-Hans' : 'en';
    return `<!doctype html>
  <html lang="${lang}">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="${escape(description)}">
    <meta name="theme-color" content="#f5f7f2">
    <title>${escape(title)} · ReproGrid</title>
    <link rel="icon" href="${assets}assets/icon.png">
    <link rel="alternate" hreflang="${alternateLang}" href="${translationHref}">
    <link rel="stylesheet" href="${assets}assets/style.css">
    <script type="module" src="${assets}assets/app.js"></script>
  </head>
  <body>
    <a class="skip" href="#main">${t('跳到内容')}</a>
    <header class="site-header">
      <div class="wrap topbar">
        ${brand(base)}
        <nav aria-label="${t('主导航')}">
          <a href="${base}#catalog"${current === 'catalog' ? ' aria-current="page"' : ''}>${t('复现目录')}</a>
          <a href="${base}methods/"${current === 'methods' ? ' aria-current="page"' : ''}>${t('评估方法')}</a>
          <a id="language-switch" class="language-switch" href="${translationHref}" lang="${alternateLang}" hreflang="${alternateLang}">${english ? '简体中文' : 'English'}</a>
        </nav>
      </div>
    </header>
    <main id="main">${content}</main>
    <footer class="site-footer"><div class="wrap footer-row">${brand(base)}
      <p>${t('面向电力电子与控制研究的独立论文复现库')}</p><a href="${base}methods/">${t('证据与收录标准')} <span aria-hidden="true">↗</span></a>
    </div></footer>
  </body></html>`;
  }

  function badge(paper) {
    const [label, style] = statuses[paper.status];
    return `<span class="status ${style}"><i aria-hidden="true"></i>${t(label)}</span>`;
  }

  function figure(source, variant, caption) {
    return `<figure class="trace ${variant}" data-trace="${source}" data-case="strong">
      <div class="trace-label"><strong>${t('有功功率阶跃响应')}</strong><span class="mono trace-case">L = 0.1 p.u.</span></div>
      <canvas aria-hidden="true"></canvas>
      <div class="trace-legend"><span><i class="line-primary"></i>RFPSC</span><span><i class="line-secondary"></i>${t('常规 PSC')}</span></div>
      <figcaption data-error="${english ? 'Chart unavailable' : '曲线暂时无法加载'}">${escape(caption)}</figcaption>
    </figure>`;
  }

  function card(paper, index) {
    const { source, title, description, domain, tags, slug } = paper;
    const venue = source.venue_policy.abbreviation;
    return `<article class="paper-card" data-paper data-domain="${domain}" data-status="${paper.status}" data-date="${escape(source.first_publication_date_openalex)}" data-citations="${source.citations.count}" data-keywords="${escape([title, source.title, description, paper.originalTitle, paper.translation.description, source.doi, source.venue, venue, ...tags].join(' '))}"${index >= 12 ? ' hidden' : ''}>
      <div class="card-meta"><span>${escape(domainTitle(domain))}</span>${badge(paper)}</div>
      <h3><a href="papers/${slug}/">${escape(title)}</a></h3>
      ${english ? '' : `<p class="paper-original" lang="en">${escape(source.title)}</p>`}
      <p class="card-description">${escape(description)}</p>
      <div class="card-bottom"><span class="mono">${escape(venue ?? 'IEEE')} · ${source.issue_year}</span>
        <a href="papers/${slug}/" aria-label="${escape(english ? 'View ' + title : '查看' + title)}">${t('查看档案')} <span aria-hidden="true">↗</span></a>
      </div>
    </article>`;
  }

  const domainButtons = Object.entries(catalog.domains).map(([id]) =>
    `<button type="button" data-domain-button="${id}" aria-pressed="false"><span>${escape(domainTitle(id))}</span><b>${entries.filter((p) => p.domain === id).length}</b></button>`
  ).join('');
  const experimentCount = entries.filter((paper) => paper.experiment).length;
  const videoCount = entries.flatMap((paper) => paper.videos).filter((video) => video.file || video.url).length;
  const homepage = page({
    base: './', title: t('论文复现库'), current: 'catalog',
    description: t('查找构网控制、PMSM 电流控制、实时仿真与数字孪生的论文复现证据。'),
    content: `<div class="wrap">
      <section class="hero" aria-labelledby="hero-title">
        <div class="hero-copy"><div class="hero-kicker"><i></i> ${t('电力电子与控制')}</div>
          <h1 id="hero-title">${t('把论文结论变成')}<br>${t('可重跑的证据')}<span>${english ? '.' : '。'}</span></h1>
          <p>${t('按领域查找论文，查看算法、开源代码、实验结果和参数缺口。')}</p>
          <div class="hero-actions"><a class="button bright" href="#catalog">${t('浏览复现目录')} <span aria-hidden="true">↓</span></a><a class="hero-link" href="papers/rfpsc/">${t('查看首篇试点')} <span aria-hidden="true">↗</span></a></div>
        </div>
        ${figure(siteBase + 'downloads/rfpsc-linear/results/chart.json', 'hero-trace', t('RFPSC 试点 · 强网空载小信号计算'))}
      </section>
      <dl class="stats">
        <div><dt>${t('收录论文')}</dt><dd>${entries.length}</dd></div>
        <div><dt>${t('研究领域')}</dt><dd>${Object.keys(catalog.domains).length}</dd></div>
        <div><dt>${t('有计算结果')}</dt><dd>${experimentCount}</dd></div>
        <div><dt>${t('讲解视频')}</dt><dd>${videoCount}</dd></div>
      </dl>
      <section id="catalog" class="catalog" aria-labelledby="catalog-title">
        <div class="section-heading"><div><h2 id="catalog-title">${t('复现目录')}</h2><p>${t('按研究方向和复现进度查找论文。')}</p></div>
          <label class="search"><i aria-hidden="true"></i><input id="search" type="search" aria-label="${t('搜索论文')}" placeholder="${t('搜索标题、算法或 DOI')}"></label>
        </div>
        <div class="catalog-layout">
          <aside class="filters" aria-label="${t('目录筛选')}">
            <h3>${t('研究领域')}</h3>
            <div class="domain-buttons">
              <button type="button" data-domain-button="all" aria-pressed="true"><span>${t('全部领域')}</span><b>${entries.length}</b></button>
              ${domainButtons}
            </div>
            <label for="status-filter">${t('复现状态')}</label>
            <select id="status-filter"><option value="all">${t('全部状态')}</option>${Object.entries(statuses).map(([key, [name]]) => `<option value="${key}">${t(name)}</option>`).join('')}</select>
          </aside>
          <div class="catalog-results">
            <div class="result-toolbar">
              <p id="match-count" role="status">${t('共')} <strong>${entries.length}</strong> ${t('篇论文')}</p>
              <label for="sort">${t('排序')} <select id="sort"><option value="curated">${t('目录顺序')}</option><option value="newest">${t('最新发表')}</option><option value="citations">${t('引用数')}</option></select></label>
            </div>
            <div id="paper-list" class="paper-list">${entries.map(card).join('')}</div>
            <div id="empty-state" class="empty" hidden><h3>${t('没有匹配的论文')}</h3><p>${t('尝试其他关键词，或清除筛选条件。')}</p><button id="reset-filters" class="button secondary" type="button">${t('清除筛选')}</button></div>
            <nav id="pagination" class="pagination" aria-label="${t('论文分页')}" hidden><button id="previous-page" type="button">${t('上一页')}</button><span id="page-label"></span><button id="next-page" type="button">${t('下一页')}</button></nav>
          </div>
        </div>
      </section>
    </div>`,
  });

  function renderExperiment(paper) {
    const prefix = `${siteBase}../../downloads/${paper.experiment}/`;
    const { report, results } = paper;
    return `<section id="algorithm" class="detail-section">
      <h2>${t('算法与复现步骤')}</h2>
      <ol class="algorithm">${report.algorithm.map(([name, description]) => `<li><strong>${escape(name)}</strong><p>${escape(description)}</p></li>`).join('')}</ol>
      <h3>${t('运行代码')}</h3>
      <pre><code>python3 run.py</code></pre>
      <div class="links-row">
        <a class="button secondary" href="${prefix}run.py" download>${t('Python 源码')} ↓</a>
        <a class="button secondary" href="${prefix}config.json" download>${t('运行配置')} ↓</a>
        <a class="button secondary" href="${prefix}LICENSE" download>${t('MIT 许可证')} ↓</a>
      </div>
    </section>
    <section id="results" class="detail-section">
      <h2>${t('复现结果')}</h2>
      <p>${t('在纯感性电网、空载与小信号条件下重算论文式 (19)。纵轴为实际功率扰动与参考阶跃之比。')}</p>
      <div class="case-controls" role="group" aria-label="${t('选择电网电感')}">
        ${results.cases.map(({ id, inductance_pu }, i) => `<button type="button" data-trace-case="${id}" data-inductance="${inductance_pu}" aria-pressed="${i === 0}">${inductance_pu.toFixed(1)} p.u.</button>`).join('')}
      </div>
      ${figure(prefix + 'results/chart.json', 'result-trace', t('0.01 p.u. 参考功率阶跃 · 50 Hz 基频'))}
      <div class="table-scroll"><table>
        <caption>${t('上升时间取首次 10% 至首次 90% 的区间，超调按峰值相对稳态值计算。')}</caption>
        <thead><tr><th scope="col">L (p.u.)</th><th scope="col">${t('PSC 上升 (ms)')}</th><th scope="col">${t('RFPSC 上升 (ms)')}</th><th scope="col">${t('PSC 超调')}</th><th scope="col">${t('RFPSC 超调')}</th></tr></thead>
        <tbody>${results.cases.map(({ inductance_pu, psc, rfpsc }) => `<tr><th scope="row">${inductance_pu.toFixed(1)}</th><td>${psc.rise_time_ms.toFixed(2)}</td><td>${rfpsc.rise_time_ms.toFixed(2)}</td><td>${psc.overshoot_percent.toFixed(2)}%</td><td>${rfpsc.overshoot_percent.toFixed(2)}%</td></tr>`).join('')}</tbody>
      </table></div>
      <p class="callout">${t('dq 模型、式 (19) 和 RFPSC 解析解通过数值交叉核对。清单记录阈值、输入与输出校验值。')}</p>
      <div class="links-row">
        <a class="button secondary" href="${prefix}results/step-response.csv" download>${t('结果 CSV')} ↓</a>
        <a class="button secondary" href="${prefix}results/summary.json" download>${t('指标 JSON')} ↓</a>
        <a class="button secondary" href="${prefix}results/manifest.json" download>${t('运行清单')} ↓</a>
      </div>
    </section>
    <section id="discussion" class="detail-section">
      <h2>${t('优点与局限')}</h2>
      <div class="discussion-grid">
        <div><h3>${t('观测到的优势')}</h3><ul>${report.strengths.map((x) => `<li>${escape(x)}</li>`).join('')}</ul></div>
        <div><h3>${t('权衡与适用条件')}</h3><ul>${report.limitations.map((x) => `<li>${escape(x)}</li>`).join('')}</ul></div>
      </div>
    </section>
    <section id="parameters" class="detail-section">
      <h2>${t('参数与待核对信息')}</h2>
      <div class="table-scroll"><table><thead><tr><th scope="col">${t('参数 / 材料')}</th><th scope="col">${t('值 / 状态')}</th><th scope="col">${t('来源 / 影响')}</th></tr></thead>
      <tbody>${report.parameters.map(([name, value, source]) => `<tr><th scope="row">${escape(name)}</th><td>${escape(value)}</td><td>${escape(source)}</td></tr>`).join('')}</tbody></table></div>
    </section>`;
  }

  function renderCandidate(paper) {
    const { source, translation } = paper;
    const testTargets = english ? source.test_targets : translation.targetsZh ?? source.test_targets;
    const boundaries = english ? source.claim_boundaries : translation.boundariesZh;
    return `<section id="algorithm" class="detail-section">
      <h2>${t('计划复核的算法与主张')}</h2>
      <p>${escape(paper.description)}</p>
      <ul>${testTargets.map((x) => `<li>${escape(x)}</li>`).join('')}</ul>
    </section>
    <section id="results" class="detail-section">
      <h2>${t('复现结果')}</h2>
      <p class="callout">${t('等待执行。此条目已整理候选信息，尚无平台运行结果、可重跑实现或实测对照。')}</p>
    </section>
    <section id="discussion" class="detail-section">
      <h2>${t('优点与局限')}</h2>
      ${boundaries?.length ? `<ul>${boundaries.map((x) => `<li>${escape(x)}</li>`).join('')}</ul>` : `<p>${t('待核对原文主张、基线和适用条件后补充。')}</p>`}
    </section>
    <section id="parameters" class="detail-section">
      <h2>${t('参数与待核对信息')}</h2>
      <p>${escape(paper.gap ?? t('参数可追溯性尚待审查。核查全文中的方程、单位、参数表和原始实验条件后再列出缺项。'))}</p>
    </section>`;
  }

  function videoTile(video, paper) {
    const prefix = `${siteBase}../../downloads/${paper.experiment}/media/`;
    const file = (value) => {
      if (!/^[\w-]+\.(mp4|webm|vtt|png|jpg)$/.test(value) || !existsSync(path.join(root, paper.folder, 'media', value))) {
        throw new Error(`Missing or invalid media file: ${value}`);
      }
      return prefix + value;
    };
    if (video.file && !paper.folder) throw new Error('Local video requires an experiment folder');
    return `<article class="video-card">
      ${video.file ? `<video controls playsinline preload="none" poster="${file(video.poster)}" aria-label="${escape(video.title)}">
        <source src="${file(video.file)}" type="${video.file.endsWith('.webm') ? 'video/webm' : 'video/mp4'}">
        ${video.captions ? `<track kind="captions" src="${file(video.captions)}" srclang="${lang}" label="${t('简体中文')}" default>` : ''}
      </video>` : ''}
      <div class="video-copy"><h3>${escape(video.title)}</h3>${video.duration ? `<span class="mono">${escape(video.duration)}</span>` : ''}<p>${escape(video.description ?? '')}</p>
      ${video.url ? external(video.url, t('打开视频')) : ''}</div>
    </article>`;
  }

  function paperPage(paper) {
    const { source } = paper;
    const anchor = [
      ['algorithm', '算法与代码'], ['results', '复现结果'], ['discussion', '优点与局限'],
      ['parameters', '参数缺口'], ['videos', '讲解视频'], ['sources', '引用来源'],
    ];
    const citation = source.citations;
    return page({
      base: '../../', route: `papers/${paper.slug}/`, title: paper.title, current: 'catalog', description: paper.description,
      content: `<div class="wrap detail">
        <a href="../../#catalog" class="breadcrumb"><span aria-hidden="true">←</span> ${t('返回复现目录')}</a>
        <header class="paper-header">
          <div class="paper-overline"><span>${escape(domainTitle(paper.domain))}</span>${badge(paper)}</div>
          <h1>${escape(paper.title)}</h1>
          ${english ? '' : `<p lang="en" class="original-title">${escape(source.title)}</p>`}
          <p class="paper-summary">${escape(paper.description)}</p>
          <div class="paper-tags">${paper.tags.map((tag) => `<span>${escape(tag)}</span>`).join('')}</div>
        </header>
        <div class="detail-layout">
          <aside class="detail-sidebar">
            <nav aria-label="${t('本页目录')}">${anchor.map(([id, name]) => `<a href="#${id}">${t(name)}</a>`).join('')}</nav>
            <dl><div><dt>${t('期刊')}</dt><dd lang="en">${escape(source.venue)}</dd></div><div><dt>${t('卷期年份')}</dt><dd>${source.issue_year}</dd></div>
              <div><dt>${t('首次发表记录')}</dt><dd>${escape(source.first_publication_date_openalex)}</dd></div>
              <div><dt>${t('OpenAlex 引用快照')}</dt><dd>${citation.count} ${t('次')} · ${escape(citation.checked_on_local)}</dd></div>
            </dl>
          </aside>
          <div class="detail-body">
            ${paper.experiment ? renderExperiment(paper) : renderCandidate(paper)}
            <section id="videos" class="detail-section"><h2>${t('讲解视频')}</h2><div class="videos">${paper.videos.map((video) => videoTile(video, paper)).join('')}</div></section>
            <section id="sources" class="detail-section"><h2>${t('引用来源')}</h2>
              <ul class="source-links"><li>${external(`https://doi.org/${source.doi}`, t('原文 DOI · ') + source.doi)}</li>
                ${source.full_text_url ? `<li>${external(source.full_text_url, t('候选全文来源'))}</li>` : ''}
                ${source.institution_record_url ? `<li>${external(source.institution_record_url, t('机构记录'))}</li>` : ''}
                ${source.reference_code_url ? `<li>${external(source.reference_code_url, t('第三方参考实现 · motulator'))}</li>` : ''}
                <li>${external(source.openalex_id, t('OpenAlex 记录'))}</li></ul>
            </section>
          </div>
        </div>
      </div>`,
    });
  }

  const methods = page({
    base: '../', route: 'methods/', title: t('评估方法'), current: 'methods', description: t('ReproGrid 的论文收录范围、复现状态和证据标准。'),
    content: `<div class="wrap methods">
      <a class="breadcrumb" href="../#catalog"><span aria-hidden="true">←</span> ${t('返回复现目录')}</a>
      <header class="paper-header"><h1>${t('评估方法')}</h1><p class="paper-summary">${t('每个结论都对应一组条件、一条可重跑的路径和可追溯的结果。')}</p></header>
      <section class="detail-section"><h2>${t('收录范围')}</h2><p>${t('目录覆盖构网控制与小信号稳定性、PMSM 电流控制、实时仿真数值方法，以及数字孪生估计。优先考虑以下 IEEE Transactions 期刊。范围外的论文需要单独说明影响或方法依据。')}</p>
        <ul class="journals">${spec.selection.journal_policy.priority_journals.map((journal) => `<li><strong>${escape(journal.abbreviation)}</strong><span lang="en">${escape(journal.title)}</span></li>`).join('')}</ul>
      </section>
      <section class="detail-section"><h2>${t('状态怎么读')}</h2><dl class="method-status">
        <div><dt>${t('试点有结果')}</dt><dd>${t('已有可执行代码和实际输出。结论的覆盖范围见各论文结果页。')}</dd></div>
        <div><dt>${t('候选待评估')}</dt><dd>${t('已登记文献信息和计划复核的主张，等待原文、参数与可执行性审查。')}</dd></div>
        <div><dt>${t('暂缓')}</dt><dd>${t('当前有关键输入缺口，例如可访问全文或必需参数。')}</dd></div>
      </dl></section>
      <section class="detail-section"><h2>${t('证据与适用边界')}</h2><ol class="algorithm">
        <li><strong>${t('记录原文条件')}</strong><p>${t('跟踪方程、参数来源、单位、标幺基准和初值。关键参数缺失时记录缺口。')}</p></li>
        <li><strong>${t('标明验证类型')}</strong><p>${t('分别标记参考代码重跑、独立实现和扩展试验。每组结果附配置、指标与运行清单。')}</p></li>
        <li><strong>${t('交叉核对与复跑')}</strong><p>${t('按模型检查解析边界、步长或容差收敛，并用独立推导或实现交叉核对。复跑结果应能重新生成指标和曲线。')}</p></li>
        <li><strong>${t('限定实验结论')}</strong><p>${t('主机运行时间、FPGA 时序、合成数据辨识和物理硬件验证分别记录。引用数按 OpenAlex 快照标注日期。')}</p></li>
      </ol></section>
    </div>`,
  });


  return { homepage, methods, paperPages: entries.map((paper) => [paper.slug, paperPage(paper)]) };
}

function write(relativePath, content) {
  const target = path.join(dist, relativePath);
  mkdirSync(path.dirname(target), { recursive: true });
  writeFileSync(target, content);
}

rmSync(dist, { recursive: true, force: true });
for (const lang of ['zh-Hans', 'en']) {
  const { homepage, methods, paperPages } = renderLanguage(lang);
  const prefix = lang === 'en' ? 'en/' : '';
  write(`${prefix}index.html`, homepage);
  write(`${prefix}methods/index.html`, methods);
  for (const [slug, html] of paperPages) write(`${prefix}papers/${slug}/index.html`, html);
}
cpSync(path.join(root, 'site'), path.join(dist, 'assets'), { recursive: true, filter: (source) => !path.basename(source).startsWith('._') });
for (const paper of papers) {
  if (paper.folder) {
    const download = path.join(dist, 'downloads', paper.experiment);
    mkdirSync(download, { recursive: true });
    for (const name of ['run.py', 'config.json', 'results', 'media']) {
      cpSync(path.join(root, paper.folder, name), path.join(download, name), {
        recursive: true, filter: (source) => !path.basename(source).startsWith('._'),
      });
    }
    cpSync(path.join(root, 'LICENSE'), path.join(download, 'LICENSE'));
  }
}
// macOS creates AppleDouble files for newly written assets on external drives.
function removeMetadata(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.name.startsWith('._') || entry.name === '.DS_Store') rmSync(filename);
    else if (entry.isDirectory()) removeMetadata(filename);
  }
}
removeMetadata(dist);
console.log(`Built ${papers.length} paper pages, in two languages.`);
