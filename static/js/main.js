function showFileName(event) {
    const input = event.target;
    const targetId = input.dataset.filenameTarget;
    if (!targetId) {
        return;
    }

    const fileNameSpan = document.getElementById(targetId);
    if (!fileNameSpan) {
        return;
    }

    fileNameSpan.textContent = input.files.length
        ? input.files[0].name
        : "Файл не выбран";
}

window.showFileName = showFileName;
