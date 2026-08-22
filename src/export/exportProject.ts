/**
 * Projekt-Export: v2-Projektdatei (kanonisches Schema) und — über den
 * Migrationsadapter — abwärtskompatible v1-Datei.
 */
import type { ProjectV2 } from "../core/model.js";
import { dumpV1 } from "../core/io/migrateV1.js";

export function toProjectJSONv2(p: ProjectV2): string { return JSON.stringify(p, null, 2); }
export function toProjectJSONv1(p: ProjectV2): string { return JSON.stringify(dumpV1(p), null, 2); }
