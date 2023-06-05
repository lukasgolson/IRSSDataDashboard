window.onload = () => {
    const ForegroundColor = ([r, g, b]) => {
        const brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
        return brightness > 0.5 ? "rgb(49, 51, 63)" : "rgb(250, 250, 250)";
    };

    const doc = window.parent.document;
    const observedElement = doc.querySelector("#root > div:nth-child(1) > div > div > div");
    const footer = doc.getElementById("newFooter");

    // Check if the elements exist, if not, exit
    if (!observedElement || !footer) return;

    let previousBgColor = window.getComputedStyle(observedElement).backgroundColor;

    const observer = new MutationObserver(() => {
        const currentBgColor = window.getComputedStyle(observedElement).backgroundColor;
        if (currentBgColor !== previousBgColor) {
            const bgColor = currentBgColor.slice(4, -1).split(', ').map(Number);
            const fontColor = ForegroundColor(bgColor);
            footer.style.color = fontColor;
            previousBgColor = currentBgColor;
        }
    });

    observer.observe(observedElement, {
        attributes: true,
        characterData: true,
        childList: true,
        subtree: true,
        attributeOldValue: true,
        characterDataOldValue: true
    });

    footer.onclick = function () {
        const randomColor = '#' + Math.floor(Math.random() * 16777215).toString(16);
        this.style.backgroundColor = randomColor;

        // Convert hex color to RGB array
        const hexToRgb = (hex) => {
            const bigint = parseInt(hex.replace('#', ''), 16);
            const r = (bigint >> 16) & 255;
            const g = (bigint >> 8) & 255;
            const b = bigint & 255;

            return [r, g, b];
        }

        this.style.color = ForegroundColor(hexToRgb(randomColor));
    };
};
