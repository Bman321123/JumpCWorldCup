document.getElementById("capture").addEventListener("click", async () => {
  const status = document.getElementById("status");
  status.textContent = "Capturing…";
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const res = await chrome.tabs.sendMessage(tab.id, { type: "pc-capture" });
    status.textContent = res && res.stored !== undefined
      ? `Logged ${res.stored} question(s).`
      : `Logger offline — run tools/crowd_server.py`;
  } catch (e) {
    status.textContent = "Open a sportspredict.com page first.";
  }
});
