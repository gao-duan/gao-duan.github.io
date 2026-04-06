(function () {
  var STORAGE_KEY = "duan-gao-blog-theme";
  var root = document.documentElement;
  var buttons = Array.prototype.slice.call(
    document.querySelectorAll("[data-theme-choice]")
  );
  var menu = document.querySelector(".theme-menu");
  var trigger = document.querySelector(".theme-menu__trigger");
  var panel = document.querySelector(".theme-menu__panel");
  var media = window.matchMedia
    ? window.matchMedia("(prefers-color-scheme: dark)")
    : null;

  function normalizePreference(value) {
    return /^(auto|light|dark)$/.test(value || "") ? value : "auto";
  }

  function currentStoredPreference() {
    var preference = root.getAttribute("data-theme-preference") || "auto";
    try {
      preference = localStorage.getItem(STORAGE_KEY) || preference;
    } catch (error) {
      preference = preference;
    }
    return normalizePreference(preference);
  }

  function resolveTheme(preference) {
    if (preference === "auto") {
      return media && media.matches ? "dark" : "light";
    }
    return preference;
  }

  function syncButtons(preference) {
    buttons.forEach(function (button) {
      var active = button.getAttribute("data-theme-choice") === preference;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    if (trigger) {
      trigger.setAttribute("aria-label", "Theme: " + preference);
    }
  }

  function setMenuOpen(open) {
    if (!menu || !trigger || !panel) {
      return;
    }
    menu.classList.toggle("is-open", open);
    panel.hidden = !open;
    trigger.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function applyPreference(preference, persist) {
    var normalized = normalizePreference(preference);
    root.setAttribute("data-theme-preference", normalized);
    root.setAttribute("data-theme", resolveTheme(normalized));
    syncButtons(normalized);
    if (!persist) {
      return;
    }
    try {
      localStorage.setItem(STORAGE_KEY, normalized);
    } catch (error) {
      return;
    }
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      applyPreference(button.getAttribute("data-theme-choice"), true);
      setMenuOpen(false);
    });
  });

  if (trigger && panel) {
    trigger.addEventListener("click", function () {
      setMenuOpen(!menu.classList.contains("is-open"));
    });

    document.addEventListener("click", function (event) {
      if (!menu || menu.contains(event.target)) {
        return;
      }
      setMenuOpen(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    });
  }

  if (media) {
    var handleSystemThemeChange = function () {
      if (currentStoredPreference() === "auto") {
        applyPreference("auto", false);
      }
    };

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", handleSystemThemeChange);
    } else if (typeof media.addListener === "function") {
      media.addListener(handleSystemThemeChange);
    }
  }

  applyPreference(currentStoredPreference(), false);
  setMenuOpen(false);
})();
