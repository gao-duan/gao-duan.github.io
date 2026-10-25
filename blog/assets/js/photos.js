(() => {
  const albums = JSON.parse(document.querySelector('#photo-data').textContent);
  const links = [...document.querySelectorAll('.photo-item')];

  document.querySelectorAll('.photo-rows').forEach(container => {
    const items = [...container.querySelectorAll('.photo-item')];
    let lastWidth = -1;
    function layout(force = false) {
      const width = container.clientWidth;
      if (!width || (!force && Math.abs(width - lastWidth) < 1)) return;
      lastWidth = width;
      const small = width <= 540, target = small ? 145 : 220;
      const gap = parseFloat(getComputedStyle(container).gap);
      const groups = [];
      let group = [], ratios = 0;
      items.forEach(item => {
        group.push(item);
        ratios += Number(item.style.getPropertyValue('--ratio'));
        if (group.length >= 2 && (
          (width - gap * (group.length - 1)) / ratios <= target * 1.2 ||
          (small && group.length === 2)
        )) {
          groups.push(group); group = []; ratios = 0;
        }
      });
      if (group.length) groups.push(group);
      const fragment = document.createDocumentFragment();
      groups.forEach((rowItems, i) => {
        const row = document.createElement('div');
        row.className = 'photo-row' + (i === groups.length - 1 ? ' photo-row--last' : '');
        rowItems.forEach(item => row.append(item));
        fragment.append(row);
      });
      container.replaceChildren(fragment);
    }
    items.forEach(item => {
      const img = item.querySelector('img');
      if (img.hasAttribute('width') && img.hasAttribute('height')) return;
      const measure = () => {
        if (img.naturalWidth && img.naturalHeight) {
          item.style.setProperty('--ratio', img.naturalWidth / img.naturalHeight);
          layout(true);
        }
      };
      img.addEventListener('load', measure);
      if (img.complete) measure();
    });
    layout();
    if (window.ResizeObserver) new ResizeObserver(() => layout()).observe(container);
    else window.addEventListener('resize', () => layout());
  });

  const dialog = document.querySelector('.photo-viewer');
  // Plain links still reach the correct image when the modal API is unavailable.
  if (!dialog || typeof dialog.showModal !== 'function') return;
  const image = document.querySelector('#viewer-image');
  const caption = document.querySelector('#viewer-caption');
  const prev = document.querySelector('.viewer-prev');
  const next = document.querySelector('.viewer-next');
  const post = document.querySelector('#viewer-post');
  const original = document.querySelector('#viewer-original');
  const lookup = new Map();
  albums.forEach(album => album.images.forEach((photo, index) => lookup.set(photo.key, {album, photo, index})));
  let selected = null, opener = null, pushed = false;
  let savedOverflow = '', savedScroll = 0;

  function render(selection) {
    selected = selection;
    const {album, photo, index} = selection;
    if (!dialog.open) {
      savedOverflow = document.documentElement.style.overflow;
      savedScroll = window.scrollY;
      dialog.showModal();
      document.documentElement.style.overflow = 'hidden';
    }
    image.alt = photo.alt;
    image.src = photo.original;
    caption.textContent = `${album.title}　${index + 1} / ${album.images.length}`;
    prev.disabled = index === 0;
    next.disabled = index === album.images.length - 1;
    post.href = photo.post;
    original.href = photo.original;
  }

  function close() {
    if (!dialog.open) return;
    dialog.close();
    document.documentElement.style.overflow = savedOverflow;
    window.scrollTo({top: savedScroll, behavior: 'instant'});
    if (opener) opener.focus({preventScroll: true});
    pushed = false;
    selected = null;
  }

  function fromLocation() {
    let key = '';
    try { key = decodeURIComponent(location.hash.slice(6)); } catch (_) { /* Invalid URL escapes. */ }
    const selection = location.hash.startsWith('#view-') && lookup.get(key);
    if (selection) render(selection);
    else close();
  }

  function requestClose() {
    if (pushed) history.back();
    else {
      history.replaceState(history.state, '', location.pathname + location.search);
      close();
    }
  }

  links.forEach(link => link.addEventListener('click', event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const selection = lookup.get(link.dataset.photoKey);
    if (!selection) return;
    event.preventDefault();
    opener = link;
    pushed = true;
    history.pushState(null, '', '#view-' + encodeURIComponent(selection.photo.key));
    render(selection);
  }));

  function step(delta) {
    if (!selected) return;
    const photo = selected.album.images[selected.index + delta];
    if (!photo) return;
    history.replaceState(history.state, '', '#view-' + encodeURIComponent(photo.key));
    render(lookup.get(photo.key));
  }
  prev.addEventListener('click', () => step(-1));
  next.addEventListener('click', () => step(1));
  document.querySelector('.viewer-close').addEventListener('click', requestClose);
  dialog.addEventListener('cancel', event => { event.preventDefault(); requestClose(); });
  dialog.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault(); step(event.key === 'ArrowLeft' ? -1 : 1);
    }
  });
  window.addEventListener('popstate', fromLocation);
  window.addEventListener('hashchange', fromLocation);
  fromLocation();
})();
