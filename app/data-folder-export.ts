export type DataFolderEntry = {
  path: string;
  contents: string;
};

export function safeDataPathSegment(value: string, fallback: string) {
  const normalized = value
    .normalize("NFKC")
    .replace(/[\\/:*?"<>|\u0000-\u001f]/g, "-")
    .replace(/\s+/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 80);
  return normalized || fallback;
}

function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function writeUint16(view: DataView, offset: number, value: number) {
  view.setUint16(offset, value, true);
}

function writeUint32(view: DataView, offset: number, value: number) {
  view.setUint32(offset, value, true);
}

/** Builds a standards-compliant, store-only ZIP so no third-party compressor is needed. */
export function createDataFolderZip(entries: DataFolderEntry[]) {
  const encoder = new TextEncoder();
  const files = entries.map((entry) => ({
    name: encoder.encode(entry.path),
    data: encoder.encode(entry.contents),
  }));
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];
  let localOffset = 0;

  for (const file of files) {
    const checksum = crc32(file.data);
    const local = new Uint8Array(30 + file.name.length + file.data.length);
    const localView = new DataView(local.buffer);
    writeUint32(localView, 0, 0x04034b50);
    writeUint16(localView, 4, 20);
    writeUint16(localView, 6, 0x0800);
    writeUint32(localView, 14, checksum);
    writeUint32(localView, 18, file.data.length);
    writeUint32(localView, 22, file.data.length);
    writeUint16(localView, 26, file.name.length);
    local.set(file.name, 30);
    local.set(file.data, 30 + file.name.length);
    localParts.push(local);

    const central = new Uint8Array(46 + file.name.length);
    const centralView = new DataView(central.buffer);
    writeUint32(centralView, 0, 0x02014b50);
    writeUint16(centralView, 4, 20);
    writeUint16(centralView, 6, 20);
    writeUint16(centralView, 8, 0x0800);
    writeUint32(centralView, 16, checksum);
    writeUint32(centralView, 20, file.data.length);
    writeUint32(centralView, 24, file.data.length);
    writeUint16(centralView, 28, file.name.length);
    writeUint32(centralView, 42, localOffset);
    central.set(file.name, 46);
    centralParts.push(central);
    localOffset += local.length;
  }

  const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0);
  const end = new Uint8Array(22);
  const endView = new DataView(end.buffer);
  writeUint32(endView, 0, 0x06054b50);
  writeUint16(endView, 8, files.length);
  writeUint16(endView, 10, files.length);
  writeUint32(endView, 12, centralSize);
  writeUint32(endView, 16, localOffset);
  return new Blob([...localParts, ...centralParts, end], { type: "application/zip" });
}
