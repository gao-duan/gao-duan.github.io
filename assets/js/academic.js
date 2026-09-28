document.addEventListener("DOMContentLoaded", function () {
  var themeStorageKey = "duan-gao-site-theme";
  var root = document.documentElement;
  var themeButtons = Array.prototype.slice.call(
    document.querySelectorAll("[data-theme-choice]")
  );
  var themeMenu = document.querySelector(".theme-menu");
  var themeTrigger = document.querySelector(".theme-menu__trigger");
  var themePanel = document.querySelector(".theme-menu__panel");
  var themeMediaQuery = window.matchMedia
    ? window.matchMedia("(prefers-color-scheme: dark)")
    : null;

  function normalizeThemePreference(value) {
    return /^(auto|light|dark)$/.test(value || "") ? value : "auto";
  }

  function currentThemePreference() {
    var preference = root.getAttribute("data-theme-preference") || "auto";
    try {
      preference = localStorage.getItem(themeStorageKey) || preference;
    } catch (error) {
      preference = preference;
    }
    return normalizeThemePreference(preference);
  }

  function resolveTheme(preference) {
    if (preference === "auto") {
      return themeMediaQuery && themeMediaQuery.matches ? "dark" : "light";
    }
    return preference;
  }

  function syncThemeButtons(preference) {
    themeButtons.forEach(function (button) {
      var active = button.getAttribute("data-theme-choice") === preference;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    if (themeTrigger) {
      themeTrigger.setAttribute("aria-label", "Theme: " + preference);
    }
  }

  function setThemeMenuOpen(open) {
    if (!themeMenu || !themeTrigger || !themePanel) {
      return;
    }
    themeMenu.classList.toggle("is-open", open);
    themePanel.hidden = !open;
    themeTrigger.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function applyThemePreference(preference, persist) {
    var normalized = normalizeThemePreference(preference);
    root.setAttribute("data-theme-preference", normalized);
    root.setAttribute("data-theme", resolveTheme(normalized));
    syncThemeButtons(normalized);
    if (!persist) {
      return;
    }
    try {
      localStorage.setItem(themeStorageKey, normalized);
    } catch (error) {
      return;
    }
  }

  themeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      applyThemePreference(button.getAttribute("data-theme-choice"), true);
      setThemeMenuOpen(false);
      if (themeTrigger) {
        themeTrigger.focus();
      }
    });
  });

  if (themeTrigger && themePanel) {
    themeTrigger.addEventListener("click", function () {
      setThemeMenuOpen(!themeMenu.classList.contains("is-open"));
    });

    document.addEventListener("click", function (event) {
      if (!themeMenu || themeMenu.contains(event.target)) {
        return;
      }
      setThemeMenuOpen(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && themeMenu.classList.contains("is-open")) {
        setThemeMenuOpen(false);
        themeTrigger.focus();
      }
    });

    themeMenu.addEventListener("focusout", function (event) {
      if (!themeMenu.contains(event.relatedTarget)) {
        setThemeMenuOpen(false);
      }
    });
  }

  if (themeMediaQuery) {
    var handleSystemThemeChange = function () {
      if (currentThemePreference() === "auto") {
        applyThemePreference("auto", false);
      }
    };

    if (typeof themeMediaQuery.addEventListener === "function") {
      themeMediaQuery.addEventListener("change", handleSystemThemeChange);
    } else if (typeof themeMediaQuery.addListener === "function") {
      themeMediaQuery.addListener(handleSystemThemeChange);
    }
  }

  applyThemePreference(currentThemePreference(), false);
  setThemeMenuOpen(false);

  var header = document.querySelector(".site-header");
  var homeLink = document.querySelector(".site-home");
  var sectionLinks = Array.prototype.slice.call(document.querySelectorAll('.site-nav__links a[href^="#"]'));
  var sections = sectionLinks.map(function (link) {
    return { link: link, section: document.getElementById(link.hash.slice(1)) };
  }).filter(function (item) { return item.section; });
  var navigationLinks = [homeLink].concat(sectionLinks).filter(Boolean);
  var pendingNavigationUpdate = false;

  function updateCurrentSection() {
    pendingNavigationUpdate = false;
    var threshold = header.getBoundingClientRect().bottom + 32;
    var current = homeLink;
    // Home and Bio can share the same scroll position on a compact header.
    if (window.scrollY > 8 || window.location.hash === "#bio") {
      sections.forEach(function (item) {
        if (item.section.getBoundingClientRect().top <= threshold) {
          current = item.link;
        }
      });
    }
    navigationLinks.forEach(function (link) {
      if (link === current) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  function scheduleNavigationUpdate() {
    if (!pendingNavigationUpdate) {
      pendingNavigationUpdate = true;
      window.requestAnimationFrame(updateCurrentSection);
    }
  }

  function updateHeaderHeight() {
    root.style.setProperty("--header-height", Math.ceil(header.getBoundingClientRect().height) + "px");
    scheduleNavigationUpdate();
  }

  if (header) {
    updateHeaderHeight();
    if (window.ResizeObserver) {
      new ResizeObserver(updateHeaderHeight).observe(header);
    }
    window.addEventListener("resize", updateHeaderHeight);
    window.addEventListener("scroll", scheduleNavigationUpdate, { passive: true });
    window.addEventListener("hashchange", scheduleNavigationUpdate);
    // Focus can move within the viewport without triggering native scrolling.
    document.addEventListener("focusin", function (event) {
      var target = event.target;
      if (header.contains(target) || target.classList.contains("skip-link")) {
        return;
      }
      window.requestAnimationFrame(function () {
        if (document.activeElement !== target) { return; }
        var top = target.getBoundingClientRect().top;
        var visibleTop = header.getBoundingClientRect().bottom + 16;
        if (top < visibleTop && window.scrollY > 0) {
          window.scrollBy({ top: top - visibleTop, behavior: "instant" });
        }
      });
    });
  }

  // Keep the initial poster clean; native controls appear only after activation.
  // Without JavaScript, the video retains its poster and native controls.
  var projectVideos = Array.prototype.slice.call(document.querySelectorAll(".project-video"));
  projectVideos.forEach(function (video) {
    var media = video.closest(".project-media");
    var poster = media.querySelector(".project-poster");
    var playButton = media.querySelector(".project-play");
    function showPoster() {
      var returnFocus = document.activeElement === video;
      video.hidden = true;
      poster.hidden = false;
      playButton.hidden = false;
      if (returnFocus) { playButton.focus({ preventScroll: true }); }
    }
    showPoster();
    playButton.addEventListener("click", function () {
      poster.hidden = true;
      playButton.hidden = true;
      video.hidden = false;
      video.focus({ preventScroll: true });
      var playback = video.play();
      if (playback && typeof playback.catch === "function") {
        playback.catch(showPoster);
      }
    });
    video.addEventListener("ended", showPoster);
  });
  var motionPreference = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  function pauseProjectVideos() {
    projectVideos.forEach(function (video) { video.pause(); });
  }
  if (motionPreference) {
    var handleMotionPreference = function () {
      if (motionPreference.matches) { pauseProjectVideos(); }
    };
    if (typeof motionPreference.addEventListener === "function") {
      motionPreference.addEventListener("change", handleMotionPreference);
    } else if (typeof motionPreference.addListener === "function") {
      motionPreference.addListener(handleMotionPreference);
    }
    handleMotionPreference();
  }
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) { pauseProjectVideos(); }
  });
});
