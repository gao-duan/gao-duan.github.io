document.addEventListener("DOMContentLoaded", function () {
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
