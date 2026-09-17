/* Progressive enhancement. Server routes remain the source of truth; every live figure comes from /api/calculate. */
(() => {
  'use strict';
  const $ = (query, root = document) => root.querySelector(query);
  const $$ = (query, root = document) => [...root.querySelectorAll(query)];
  const csrf = document.body.dataset.csrf || '';
  let toastTimer;
  function toast(message) {
    const el = $('#toast'); if (!el) return;
    clearTimeout(toastTimer); el.textContent = message; el.hidden = false;
    toastTimer = setTimeout(() => { el.hidden = true; }, 3800);
  }
  // Navigation drawer
  const menu = $('.menu-toggle'), sidebar = $('.sidebar'), scrim = $('.sidebar-scrim');
  function closeMenu() {
    sidebar?.classList.remove('open'); menu?.setAttribute('aria-expanded', 'false');
    if (scrim) scrim.hidden = true;
    document.body.classList.remove('no-scroll');
  }
  menu?.addEventListener('click', () => {
    const opened = sidebar.classList.toggle('open');
    menu.setAttribute('aria-expanded', String(opened)); scrim.hidden = !opened;
    document.body.classList.toggle('no-scroll', opened);
    if (opened) $('a', sidebar)?.focus();
  });
  scrim?.addEventListener('click', () => { closeMenu(); menu.focus(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && sidebar?.classList.contains('open')) { closeMenu(); menu.focus(); }
    if (event.key === 'Tab' && sidebar?.classList.contains('open')) {
      const items = $$('a, button', sidebar); const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  matchMedia('(min-width:741px)').addEventListener('change', e => { if (e.matches) closeMenu(); });
  // Small helpers
  $('[data-password-toggle]')?.addEventListener('click', event => {
    const input = $('#password'); const show = input.type === 'password';
    input.type = show ? 'text' : 'password'; event.currentTarget.textContent = show ? 'Hide' : 'Show';
    event.currentTarget.setAttribute('aria-pressed', String(show));
  });
  $('[data-print]')?.addEventListener('click', () => window.print());
  $$('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const node = document.getElementById(button.dataset.copy); const text = node.textContent;
    try {
      if (!navigator.clipboard) throw new Error('Clipboard not available');
      await navigator.clipboard.writeText(text); toast('Corrected letter copied to clipboard.');
    } catch {
      const range = document.createRange(); range.selectNodeContents(node);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      toast('Text selected. Use Ctrl+C or Command+C to copy.');
    }
  }));
  const alert = $('.alert-error[role=alert]');
  if (alert && !$('[data-loan-form]')) alert.focus();
  $$('form[data-busy-on-submit]').forEach(f => f.addEventListener('submit', event => {
    if (f.dataset.busy) { event.preventDefault(); return; }
    f.dataset.busy = '1'; f.setAttribute('aria-busy', 'true'); f.classList.add('form-busy');
    const button = f.querySelector('button:not([type=button])');
    if (button) { button.disabled = true; button.dataset.label = button.textContent; button.textContent = 'Working…'; }
  }));
  window.addEventListener('pageshow', () => $$('form[data-busy-on-submit]').forEach(f => {
    delete f.dataset.busy; f.removeAttribute('aria-busy'); f.classList.remove('form-busy');
    const button = f.querySelector('button:not([type=button])');
    if (button && button.dataset.label) { button.disabled = false; button.textContent = button.dataset.label; }
  }));
  // Decision form: reveal the override explanation when the choice differs from the recommendation
  const decision = $('[data-decision-form]');
  if (decision) {
    const recommended = decision.dataset.recommended;
    const overrideField = $('[data-override-field]', decision), amountField = $('[data-amount-field]', decision);
    const sync = () => {
      const chosen = decision.elements.human_decision.value;
      const differs = chosen && chosen !== 'manual_review' && recommended && chosen !== recommended;
      overrideField.hidden = !differs;
      overrideField.querySelector('textarea').required = !!differs;
      amountField.hidden = !(chosen === 'approved' || chosen === 'conditionally_approved');
    };
    decision.addEventListener('change', sync); sync();
  }
  // Step form with server-side calculations
  const form = $('[data-loan-form]'); if (!form) return;
  const steps = $$('[data-step]', form), tabs = $$('[data-goto-step]', form);
  const next = $('[data-next]', form), previous = $('[data-previous]', form), submit = $('[data-submit]', form), saveDraft = $('[data-save-draft]', form);
  const status = $('[data-calc-status]'), flags = $('[data-live-flags]');
  let current = 0, busy = false, calcTimer, lastPayload = '';
  const money = v => Number.isFinite(v) ? new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', maximumFractionDigits: 0}).format(v) : '—';
  const pct = v => Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—';
  const number = name => { const raw = form.elements[name]?.value ?? ''; return raw.trim() ? Number(raw) : NaN; };
  function payload() {
    const data = {};
    new FormData(form).forEach((value, key) => {
      if (key === 'csrf_token' || key === 'action') return;
      if (key === 'documents') { (data.documents ||= []).push(value); return; }
      if (String(value).trim() !== '') data[key] = value;
    });
    data.documents ||= [];
    return data;
  }
  function render(derived) {
    const map = {monthly_income: money(derived.monthly_income) + (derived.income_basis ? ` (${derived.income_basis})` : ''),
      gross_equity: money(derived.gross_equity), available_equity: money(derived.available_equity), cltv_after: pct(derived.cltv_after),
      dti_after: pct(derived.dti_after), estimated_payment: money(derived.estimated_payment) + (derived.payment_basis ? ` (${derived.payment_basis})` : ''),
      reserves_months: Number.isFinite(derived.reserves_months) ? derived.reserves_months.toFixed(1) : '—', doc_completeness: pct(derived.doc_completeness)};
    $$('[data-live]').forEach(el => { el.textContent = map[el.dataset.live] ?? '—'; });
    const value = number('property_value'), mortgage = number('mortgage_balance'), loan = number('requested_amount');
    const m = value > 0 && Number.isFinite(mortgage) ? Math.max(0, Math.min(100, mortgage / value * 100)) : 0;
    const l = value > 0 && Number.isFinite(loan) ? Math.max(0, Math.min(100 - m, loan / value * 100)) : 0;
    $('[data-equity-mortgage]').style.width = `${m}%`; $('[data-equity-loan]').style.width = `${l}%`;
    flags.replaceChildren(...(derived.flags || []).map(f => { const li = document.createElement('li'); li.className = `flag flag-${f.severity}`; li.textContent = `${f.code} · ${f.message}`; return li; }));
  }
  async function calculate() {
    const data = payload(); const body = JSON.stringify(data);
    if (body === lastPayload) return; lastPayload = body;
    if (status) status.textContent = 'calculating…';
    try {
      const response = await fetch(form.dataset.calcUrl, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body});
      const result = await response.json();
      if (result.ok) { render(result.derived); if (status) status.textContent = 'from the server'; }
      else { $$('[data-live]').forEach(el => { el.textContent = '—'; }); flags.replaceChildren(); if (status) status.textContent = `waiting for: ${Object.keys(result.errors || {}).slice(0, 3).join(', ').replaceAll('_', ' ')}`; }
    } catch { if (status) status.textContent = 'unavailable'; }
  }
  function scheduleCalc() { clearTimeout(calcTimer); calcTimer = setTimeout(calculate, 350); }
  function validStep(index) {
    const invalid = $$('input, select', steps[index]).find(el => !el.checkValidity());
    if (invalid) { showStep(index, false); invalid.reportValidity(); return false; }
    return true;
  }
  function review() {
    const list = $('#application-review'); if (!list) return; list.replaceChildren();
    $$('input:not([type=checkbox]):not([type=hidden]), select', form).forEach(input => {
      const label = input.closest('label')?.firstChild?.textContent?.trim(); if (!label) return;
      const group = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = label;
      const raw = input.tagName === 'SELECT' ? input.selectedOptions[0]?.textContent : input.value;
      dd.textContent = raw ? (input.type === 'number' && label.includes('$') ? money(Number(raw)) : raw) : 'Not provided';
      group.append(dt, dd); list.append(group);
    });
    const docs = $$('input[name=documents]:checked', form).map(i => i.nextElementSibling.textContent);
    const group = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = 'Documents'; dd.textContent = docs.length ? docs.join(', ') : 'None ticked'; group.append(dt, dd); list.append(group);
  }
  function showStep(index, focus = true) {
    current = index;
    steps.forEach((step, i) => { step.hidden = i !== index; });
    tabs.forEach((tab, i) => { tab.classList.toggle('active', i === index); tab.classList.toggle('complete', i < index); if (i === index) tab.setAttribute('aria-current', 'step'); else tab.removeAttribute('aria-current'); });
    previous.hidden = index === 0; next.hidden = index === steps.length - 1; submit.hidden = index !== steps.length - 1;
    $('.step-caption', form).textContent = `Step ${index + 1} of ${steps.length}`;
    if (focus) { const heading = $('h2', steps[index]); heading.tabIndex = -1; heading.focus({preventScroll: true});
      if (matchMedia('(max-width:740px)').matches) $('.stepper', form).scrollIntoView({block: 'start', behavior: matchMedia('(prefers-reduced-motion:reduce)').matches ? 'instant' : 'smooth'}); }
    review();
  }
  next.addEventListener('click', () => { if (validStep(current)) showStep(current + 1); });
  previous.addEventListener('click', () => showStep(Math.max(0, current - 1)));
  tabs.forEach(tab => tab.addEventListener('click', () => {
    const target = Number(tab.dataset.gotoStep);
    for (let i = 0; i < target; i++) { if (!validStep(i)) return; }
    showStep(target);
  }));
  form.addEventListener('input', scheduleCalc); form.addEventListener('change', () => { scheduleCalc(); review(); });
  form.noValidate = true;
  form.addEventListener('submit', event => {
    if (busy) { event.preventDefault(); return; }
    if (event.submitter === saveDraft) { busy = true; return; }               // drafts skip validation
    if (current < steps.length - 1) { event.preventDefault(); if (validStep(current)) showStep(current + 1); return; }
    for (let i = 0; i < steps.length; i++) { if (!validStep(i)) { event.preventDefault(); return; } }
    busy = true; submit.disabled = true; previous.disabled = true; tabs.forEach(tab => tab.disabled = true);
    form.setAttribute('aria-busy', 'true'); $('.submit-status', form).hidden = false;
  });
  window.addEventListener('pageshow', () => { busy = false; submit.disabled = false; previous.disabled = false; tabs.forEach(tab => tab.disabled = false); form.removeAttribute('aria-busy'); $('.submit-status', form).hidden = true; });
  showStep(0, false);
  const invalid = $('[aria-invalid=true]', form);
  if (invalid) { showStep(Number(invalid.closest('[data-step]').dataset.step), false); invalid.focus(); }
  else if (alert) { alert.focus(); }
  calculate();
})();
