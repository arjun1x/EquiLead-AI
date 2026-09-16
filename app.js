/* Progressive enhancement. Server routes remain the source of truth. */
(() => {
  'use strict';
  const $ = (query, root = document) => root.querySelector(query);
  const $$ = (query, root = document) => [...root.querySelectorAll(query)];
  let toastTimer;
  function toast(message) {
    const el = $('#toast'); if (!el) return;
    clearTimeout(toastTimer); el.textContent = message; el.hidden = false;
    toastTimer = setTimeout(() => { el.hidden = true; }, 3800);
  }
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
  $('[data-password-toggle]')?.addEventListener('click', event => {
    const input = $('#password'); const show = input.type === 'password';
    input.type = show ? 'text' : 'password'; event.currentTarget.textContent = show ? 'Hide' : 'Show';
    event.currentTarget.setAttribute('aria-pressed', String(show));
  });
  $('[data-print]')?.addEventListener('click', () => window.print());
  $$('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy).textContent;
    try {
      if (!navigator.clipboard) throw new Error('Clipboard not available');
      await navigator.clipboard.writeText(text); toast('Draft copied to clipboard.');
    } catch {
      const node = document.getElementById(button.dataset.copy);
      const range = document.createRange(); range.selectNodeContents(node);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      toast('Text selected. Use Ctrl+C or Command+C to copy.');
    }
  }));
  // Server-side errors: move focus to the summary so keyboard and screen-reader users land on it.
  const alert = $('.alert-error[role=alert]');
  if (alert && !$('[data-loan-form]')) alert.focus();
  // Simple forms: show a busy state once, and never submit twice.
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
  const form = $('[data-loan-form]'); if (!form) return;
  const steps = $$('[data-step]', form), tabs = $$('[data-goto-step]', form);
  const next = $('[data-next]', form), previous = $('[data-previous]', form), submit = $('[data-submit]', form);
  let current = 0, busy = false;
  const money = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0}).format(value) : '—';
  const number = name => { const raw = form.elements[name].value; return raw.trim() ? Number(raw) : NaN; };
  function validStep(index) {
    const invalid = $$('input, select', steps[index]).find(el => !el.checkValidity());
    if (invalid) { showStep(index, false); invalid.reportValidity(); return false; }
    return true;
  }
  const labelMap = {applicant_name:'Applicant',loan:'Requested loan',reason:'Loan purpose',value:'Property value',mortdue:'Mortgage balance',job:'Occupation',yoj:'Employment years',debtinc:'Debt-to-income (%)',clage:'Credit history (months)',clno:'Credit lines',ninq:'Recent inquiries',derog:'Derogatory reports',delinq:'Delinquent lines'};
  function update() {
    const loan = number('loan'), mortgage = number('mortdue'), value = number('value');
    const equity = value - mortgage - loan, ratio = value > 0 ? (mortgage + loan) / value * 100 : NaN;
    const amounts = {loan:money(loan),mortdue:money(mortgage),equity:money(equity),ltv:Number.isFinite(ratio)?`${ratio.toFixed(1)}%`:'—'};
    $$('[data-summary]', form).forEach(el => { el.textContent = amounts[el.dataset.summary]; });
    const m = value>0 && Number.isFinite(mortgage) ? Math.max(0,Math.min(100,mortgage/value*100)):0;
    const l = value>0 && Number.isFinite(loan) ? Math.max(0,Math.min(100-m,loan/value*100)):0;
    $('[data-equity-mortgage]',form).style.width = `${m}%`; $('[data-equity-loan]',form).style.width = `${l}%`;
    const review = $('#application-review'); review.replaceChildren();
    Object.entries(labelMap).forEach(([name,label]) => {
      const input = form.elements[name]; const group=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');
      dt.textContent=label;
      dd.textContent=input.tagName==='SELECT'?input.selectedOptions[0]?.textContent:(['loan','mortdue','value'].includes(name)?money(number(name)):input.value||'Not provided');
      group.append(dt,dd); review.append(group);
    });
  }
  function showStep(index, focus=true) {
    current=index;
    steps.forEach((step,i)=>{step.hidden=i!==index;});
    tabs.forEach((tab,i)=>{tab.classList.toggle('active',i===index);tab.classList.toggle('complete',i<index); if(i===index)tab.setAttribute('aria-current','step');else tab.removeAttribute('aria-current');});
    previous.hidden=index===0; next.hidden=index===steps.length-1; submit.hidden=index!==steps.length-1;
    $('.step-caption',form).textContent=`Step ${index+1} of ${steps.length}`;
    if(focus){const heading=$('h2',steps[index]); heading.tabIndex=-1; heading.focus({preventScroll:true});
      if(matchMedia('(max-width:740px)').matches) $('.stepper',form).scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});
    }
    update();
  }
  next.addEventListener('click',()=>{if(validStep(current))showStep(current+1);});
  previous.addEventListener('click',()=>showStep(Math.max(0,current-1)));
  tabs.forEach(tab=>tab.addEventListener('click',()=>{
    const target=Number(tab.dataset.gotoStep);
    for(let i=0;i<target;i++){if(!validStep(i))return;}
    showStep(target);
  }));
  form.addEventListener('input',update); form.addEventListener('change',update);
  // Native validation would try to focus a hidden control; reveal the correct step ourselves.
  form.noValidate=true;
  form.addEventListener('submit',event=>{
    if(busy){event.preventDefault();return;}
    if(current<steps.length-1){event.preventDefault();if(validStep(current))showStep(current+1);return;}
    for(let i=0;i<steps.length;i++){if(!validStep(i)){event.preventDefault();return;}}
    busy=true; submit.disabled=true; previous.disabled=true; tabs.forEach(tab=>tab.disabled=true);
    form.setAttribute('aria-busy','true'); $('.submit-status',form).hidden=false;
  });
  window.addEventListener('pageshow',()=>{busy=false;submit.disabled=false;previous.disabled=false;tabs.forEach(tab=>tab.disabled=false);form.removeAttribute('aria-busy');$('.submit-status',form).hidden=true;});
  showStep(0,false);
  const invalid=$('[aria-invalid=true]',form);
  if(invalid){showStep(Number(invalid.closest('[data-step]').dataset.step),false);invalid.focus();}
  else if(alert){alert.focus();}
})();
