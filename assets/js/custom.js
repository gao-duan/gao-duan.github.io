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
      if (event.key === "Escape") {
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

  var lastModified = document.getElementById("lastModified");
  if (!lastModified) {
    return;
  }

  var modifiedDate = new Date(document.lastModified);
  if (Number.isNaN(modifiedDate.getTime())) {
    lastModified.textContent = "";
    return;
  }

  lastModified.textContent =
    "Last modified on " +
    modifiedDate.getFullYear() +
    "-" +
    String(modifiedDate.getMonth() + 1).padStart(2, "0") +
    "-" +
    String(modifiedDate.getDate()).padStart(2, "0");
});
