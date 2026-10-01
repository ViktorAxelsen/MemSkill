/* MemSkill project page. Static HTML is the source of truth; JS adds interaction. */
(() => {
  'use strict';

  const menu = document.querySelector('#site-nav');
  const menuToggle = document.querySelector('.menu-toggle');
  function closeMenu(restoreFocus = false) {
    menu.classList.remove('is-open');
    menuToggle.setAttribute('aria-expanded', 'false');
    menuToggle.setAttribute('aria-label', 'Open navigation');
    if (restoreFocus) menuToggle.focus();
  }
  menuToggle.addEventListener('click', () => {
    const open = menu.classList.toggle('is-open');
    menuToggle.setAttribute('aria-expanded', String(open));
    menuToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  });
  menu.addEventListener('click', (event) => {
    if (event.target.closest('a')) closeMenu();
  });
  document.addEventListener('click', (event) => {
    if (!event.target.closest('.site-header')) closeMenu();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && menu.classList.contains('is-open')) closeMenu(true);
  });
  window.matchMedia('(min-width: 761px)').addEventListener('change', () => closeMenu());

  // One animation frame per scroll, with no continuously running render loop.
  const progress = document.querySelector('.reading-progress');
  const sectionLinks = [...menu.querySelectorAll('a')];
  const sections = sectionLinks.map((link) => document.querySelector(link.hash));
  let scrollPending = false;
  function updateScroll() {
    const distance = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.transform = `scaleX(${distance > 0 ? Math.min(1, Math.max(0, window.scrollY / distance)) : 0})`;
    let current = -1;
    sections.forEach((section, index) => {
      if (section.getBoundingClientRect().top <= 160) current = index;
    });
    sectionLinks.forEach((link, index) => {
      if (index === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    scrollPending = false;
  }
  function scheduleScroll() {
    if (!scrollPending) {
      scrollPending = true;
      window.requestAnimationFrame(updateScroll);
    }
  }
  window.addEventListener('scroll', scheduleScroll, { passive: true });
  window.addEventListener('resize', scheduleScroll);
  window.addEventListener('load', scheduleScroll);
  updateScroll();

  // The hero is an explanatory diagram, not a simulated live agent run.
  const descriptions = {
    select: 'The controller learns to select relevant skills for each text span and its retrieved memories.',
    apply: 'The executor composes the selected skills in one LLM call to construct structured memory updates.',
    evolve: 'The designer learns from representative hard cases to refine existing skills and propose new ones.'
  };
  const stageButtons = [...document.querySelectorAll('[data-stage]')];
  stageButtons.forEach((button) => {
    button.addEventListener('click', () => {
      stageButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      document.querySelectorAll('[data-node]').forEach((node) => {
        node.classList.toggle('is-active', node.dataset.node === button.dataset.stage);
      });
      document.querySelector('#stage-description').textContent = descriptions[button.dataset.stage];
    });
  });

  // Accessible tab groups: roving tabindex and Arrow/Home/End keyboard navigation.
  document.querySelectorAll('[data-tab-group]').forEach((group) => {
    const tabs = [...group.querySelectorAll('[role="tab"]')];
    function activate(tab, focus = false) {
      tabs.forEach((item) => {
        const selected = item === tab;
        item.setAttribute('aria-selected', String(selected));
        item.tabIndex = selected ? 0 : -1;
        document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
      });
      if (focus) tab.focus();
      scheduleScroll();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => activate(tab));
      tab.addEventListener('keydown', (event) => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          activate(tabs[next], true);
        }
      });
    });
  });

  // Source: arXiv:2602.02474v2, Figure 3 (LLaMA, LoCoMo-to-HotpotQA transfer).
  const transferScores = {
    7: { 50: 67.57, 100: 71.48, 200: 67.97 },
    5: { 50: 64.85, 100: 68.36, 200: 62.89 },
    3: { 50: 64.06, 100: 68.75, 200: 59.76 }
  };
  document.querySelector('#skill-budget').addEventListener('change', (event) => {
    const scores = transferScores[event.target.value];
    document.querySelectorAll('.chart-bar[data-context]').forEach((bar) => {
      const score = scores[bar.dataset.context].toFixed(2);
      bar.style.setProperty('--value', score);
      // Preserve the accessible series label when updating the visible value.
      bar.querySelector('span').lastChild.textContent = score;
    });
    document.querySelector('#chart-announcement').textContent =
      `With ${event.target.value} skills: 50 documents ${scores[50].toFixed(2)}, ` +
      `100 documents ${scores[100].toFixed(2)}, 200 documents ${scores[200].toFixed(2)}.`;
  });

  // Links still open the original image when dialog is unavailable or JS is disabled.
  const dialog = document.querySelector('#figure-dialog');
  const image = document.querySelector('#dialog-image');
  const imageWrap = document.querySelector('.dialog-image-wrap');
  const zoomButton = document.querySelector('#figure-zoom');
  if (typeof dialog.showModal === 'function') {
    document.querySelectorAll('[data-figure]').forEach((link) => {
      link.addEventListener('click', (event) => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        image.src = link.href;
        image.alt = link.querySelector('img')?.alt || link.dataset.title;
        document.querySelector('#dialog-title').textContent = link.dataset.title;
        document.querySelector('#figure-original').href = link.href;
        imageWrap.classList.remove('is-zoomed');
        zoomButton.setAttribute('aria-pressed', 'false');
        zoomButton.textContent = 'Actual size';
        dialog.showModal();
        imageWrap.scrollTo(0, 0);
        document.body.classList.add('modal-open');
      });
    });
    dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (event) => {
      const bounds = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
    });
    dialog.addEventListener('close', () => document.body.classList.remove('modal-open'));
    zoomButton.addEventListener('click', () => {
      const zoomed = imageWrap.classList.toggle('is-zoomed');
      zoomButton.setAttribute('aria-pressed', String(zoomed));
      zoomButton.textContent = zoomed ? 'Fit to view' : 'Actual size';
      imageWrap.scrollTo(0, 0);
    });
  }

  // Clipboard API on HTTPS/localhost; selection fallback also works with file://.
  const copyButton = document.querySelector('#copy-citation');
  const copyStatus = document.querySelector('#copy-status');
  let copyReset;
  copyButton.addEventListener('click', async () => {
    const code = document.querySelector('#bibtex');
    let copied = false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code.textContent.trim());
        copied = true;
      }
    } catch { /* Try the selection fallback below. */ }
    if (!copied) {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      try { copied = document.execCommand('copy'); } catch { copied = false; }
      if (copied) selection.removeAllRanges();
    }
    clearTimeout(copyReset);
    copyButton.querySelector('span').textContent = copied ? 'Copied!' : 'Citation selected';
    copyStatus.textContent = copied ? 'Citation copied to clipboard.' : 'Press Ctrl+C (Windows) or Command+C (Mac) to copy the selected citation.';
    copyStatus.hidden = false;
    copyReset = setTimeout(() => {
      copyButton.querySelector('span').textContent = 'Copy citation';
      copyStatus.hidden = true;
    }, 5000);
  });
})();
