import { createRoot } from "react-dom/client";
import MemoryGuardApp from "../app/page";
import "../app/globals.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Desktop application root was not found.");
}

createRoot(rootElement).render(<MemoryGuardApp />);
