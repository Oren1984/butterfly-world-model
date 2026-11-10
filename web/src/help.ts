/**
 * Help Center: a modal dialog explaining the project, its three modes and how to read the screen.
 * The content is static HTML in index.html; this module only opens, closes and navigates it.
 */

const dialog = document.getElementById("help") as HTMLDialogElement;
const openBtn = document.getElementById("btn-help") as HTMLButtonElement;
const closeBtn = document.getElementById("btn-help-close") as HTMLButtonElement;

openBtn.addEventListener("click", () => {
  dialog.showModal();
  dialog.querySelector(".help-body")?.scrollTo({ top: 0 });
});

closeBtn.addEventListener("click", () => dialog.close());

// A click on the backdrop (outside the dialog box) closes it. Esc is handled by <dialog> itself.
dialog.addEventListener("click", (ev) => {
  if (ev.target === dialog) dialog.close();
});

dialog.addEventListener("close", () => openBtn.focus());

// Topic links scroll inside the dialog instead of changing the URL hash, which selects the mode.
for (const a of Array.from(dialog.querySelectorAll<HTMLAnchorElement>(".help-toc a"))) {
  a.addEventListener("click", (ev) => {
    ev.preventDefault();
    document.getElementById(a.getAttribute("href")!.slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}
