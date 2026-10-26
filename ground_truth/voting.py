LABELS = {"IZQUIERDA", "CENTRO", "DERECHA"}
IZQ    = "IZQUIERDA"
CEN    = "CENTRO"
DER    = "DERECHA"


def normalize_label(label) -> str | None:
    if not isinstance(label, str):
        return None
    mapping = {
        "LEFT": IZQ, "IZQUIERDA": IZQ,
        "CENTER": CEN, "CENTRE": CEN, "CENTRO": CEN,
        "RIGHT": DER, "DERECHA": DER,
    }
    return mapping.get(label.upper().strip())


def apply_voting(claude_label, gpt_label, llama_label) -> dict:
    c  = claude_label if claude_label in LABELS else "ERROR"
    g  = gpt_label    if gpt_label    in LABELS else "ERROR"
    ll = llama_label  if llama_label  in LABELS else "ERROR"

    # 3/3 acuerdo
    if c == g == ll == IZQ:
        return _v(IZQ, "ALTA",   "INCLUIR",  "3/3 acuerdo IZQUIERDA — alta confianza.")
    if c == g == ll == DER:
        return _v(DER, "ALTA",   "INCLUIR",  "3/3 acuerdo DERECHA — alta confianza.")
    if c == g == ll == CEN:
        return _v(CEN, "BAJA",   "INCLUIR",  "3/3 CENTRO — Llama (sesgo LEFT) también dice CEN, señal de artículo genuinamente moderado.")

    # 2/3 IZQUIERDA
    if c == IZQ and g == IZQ and ll == CEN:
        return _v(IZQ, "BAJA",   "INCLUIR",  "Claude+GPT IZQ, Llama CEN. Mayoría 2/3 IZQUIERDA — confianza BAJA por discrepancia con Llama.")
    if c == IZQ and g == IZQ and ll == DER:
        return _v(IZQ, "BAJA",   "INCLUIR",  "Claude+GPT IZQ, Llama DER. Mayoría 2/3 IZQUIERDA — confianza BAJA por anomalía de Llama contra su sesgo LEFT.")

    # 2/3 DERECHA
    if c == DER and g == DER and ll == CEN:
        return _v(DER, "ALTA",   "INCLUIR",  "Claude+GPT DER, Llama CEN. ESPERADO: sesgo LEFT de Llama colapsa a CEN ante texto de derecha (Kennedy et al. 2025).")
    if c == DER and g == DER and ll == IZQ:
        return _v(DER, "BAJA",   "INCLUIR",  "Claude+GPT DER, Llama IZQ. Mayoría 2/3 DERECHA — Llama IZQ coherente con su sesgo LEFT, no descarta la etiqueta.")

    # Claude+GPT CENTRO vs Llama
    if c == CEN and g == CEN and ll == IZQ:
        return _v(CEN, "BAJA",   "REVISAR",  "Claude+GPT CEN, Llama IZQ. Ambiguo: sesgo LEFT de Llama vs center-collapse comercial. Revisar manualmente.")
    if c == CEN and g == CEN and ll == DER:
        return _v(CEN, "MEDIA",  "INCLUIR",  "Claude+GPT CEN, Llama DER. Inusual dado sesgo LEFT de Llama — si dice DER el artículo probablemente no es IZQ. Se acepta CEN.")

    # Mezclas con 1 CENTRO
    if c == IZQ and g == CEN and ll == IZQ:
        return _v(IZQ, "MEDIA",  "INCLUIR",  "Claude+Llama IZQ, GPT CEN. Mayoría 2/3 IZQUIERDA.")
    if c == DER and g == CEN and ll == DER:
        return _v(DER, "MEDIA",  "INCLUIR",  "Claude+Llama DER, GPT CEN. Mayoría 2/3 DERECHA.")
    if c == CEN and g == IZQ and ll == IZQ:
        return _v(IZQ, "MEDIA",  "INCLUIR",  "GPT+Llama IZQ, Claude CEN. Mayoría 2/3 IZQUIERDA.")
    if c == CEN and g == DER and ll == DER:
        return _v(DER, "MEDIA",  "INCLUIR",  "GPT+Llama DER, Claude CEN. Mayoría 2/3 DERECHA.")

    # Casos sin CENTRO
    if c == IZQ and g == DER and ll == IZQ:
        return _v(IZQ, "MEDIA",  "INCLUIR",  "Claude+Llama IZQ, GPT DER. GPT es el outlier. Mayoría 2/3 IZQUIERDA.")
    if c == DER and g == IZQ and ll == DER:
        return _v(DER, "BAJA",   "INCLUIR",  "Claude+Llama DER, GPT IZQ. Llama dice DER contra su sesgo LEFT — señal fuerte. Mayoría 2/3 DERECHA.")
    if c == IZQ and g == DER and ll == DER:
        return _v(DER, "BAJA",   "INCLUIR",  "GPT+Llama DER, Claude IZQ. Llama dice DER contra su sesgo LEFT — señal fuerte. Mayoría 2/3 DERECHA.")
    if c == DER and g == IZQ and ll == IZQ:
        return _v(IZQ, "MEDIA",  "INCLUIR",  "GPT+Llama IZQ, Claude DER. Mayoría 2/3 IZQUIERDA.")

    # 2/3 CENTRO (con un outlier que no es el par Claude+GPT)
    if c == IZQ and g == CEN and ll == CEN:
        return _v(CEN, "BAJA",  "REVISAR", "GPT+Llama CEN, Claude IZQ. Llama (sesgo IZQ) dice CEN — probable artículo moderado. Revisar center-collapse.")
    if c == CEN and g == IZQ and ll == CEN:
        return _v(CEN, "BAJA",  "REVISAR", "Claude+Llama CEN, GPT IZQ. 2/3 CENTRO. Revisar center-collapse.")
    if c == CEN and g == DER and ll == CEN:
        return _v(DER, "BAJA",  "INCLUIR",  "GPT DER, Claude+Llama CEN. GPT superó su center-collapse; Llama (sesgo LEFT) dice CEN en lugar de IZQ. Claude sigue en CEN — confianza BAJA pero se acepta DER.")
    if c == DER and g == CEN and ll == CEN:
        return _v(DER, "MEDIA", "INCLUIR",  "Claude DER, GPT+Llama CEN. Claude superó su center-collapse; Llama (sesgo LEFT) dice CEN en lugar de IZQ — señal de que el artículo no es IZQ. Se acepta DER con confianza MEDIA.")

    # Llama con error — fallback a acuerdo Claude+GPT
    if ll == "ERROR":
        if c == g and c in LABELS:
            return _v(c, "MEDIA", "INCLUIR",
                      f"Claude+GPT acuerdan {c}, Llama con error. Aceptado con confianza MEDIA.")
        if c in LABELS and g in LABELS:
            return _v(None, "NINGUNA", "REVISAR",
                      f"Claude={c}, GPT={g}, Llama con error. Sin mayoría clara. Revisar.")

    # Desacuerdo total
    return _v(None, "NINGUNA", "DESCARTAR",
              f"Desacuerdo total: Claude={c}, GPT={g}, Llama={ll}. Descartado.")


def _v(label, confianza, accion, razonamiento):
    return {"final_label": label, "confianza": confianza,
            "accion": accion, "razonamiento_voto": razonamiento}
