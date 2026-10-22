"""Write Spanish CPU evidence summary from actual frozen artifacts; no HIP execution."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
repo=Path(__file__).resolve().parents[1];data=repo/'reports/tp-decode-private-data'
raw=Path('external-data/work/tp-decode-v1-build');attempt='host-r2'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
nodes={n:load(data/n/attempt/'handoff.json') for n in ('node1','node2')}
assert nodes['node1']['source_files']==nodes['node2']['source_files']
assert nodes['node1']['build']['binary_sha256']==nodes['node2']['build']['binary_sha256']
assert load(data/'cpu-validation.json')['passed']
for node in nodes:
    r=load(data/node/attempt/'resources.json');assert r['actual_new_functions']==30 and r['old11_byte_identity']=='PASS'
    evidence=[]
    for row in range(1,9):
        text=(raw/node/attempt/f'B-R{row}.isa').read_text();start=text.index('s_waitcnt vmcnt(3)');end=text.index('s_waitcnt vmcnt(0)',start)
        # Retain actual instructions around loads/waits/hash/dot, not a source claim.
        lines=text[:end].splitlines();selected=[line for line in lines if re.search('global_load_b32|s_waitcnt vmcnt|v_pk_mul_lo_u16|dot2',line)]
        evidence.append(dict(row=row,ISA_sha256=sha(raw/node/attempt/f'B-R{row}.isa'),actual_excerpt=selected,
                             conclusion='Orden estático RAW→vmcnt(3)→hash/dot actual antes del wait completo; latencia medida OPEN'))
    save(data/node/'isa-proof.json',dict(cpu_only=True,source_is_not_latency_proof=True,functions=evidence))
save(data/'handoff.json',dict(state='CPU_BUILD_PASS_GPU_OPEN',branch='performance/tp-decode-v1',base='205ad212c0c311be72ccad88ae5dd6106cf3690e',
    nodes={n:dict(scratch=x['scratch'],binary_sha256=x['build']['binary_sha256'],control_archive_sha256=x['build']['archive_sha256'],
                  frozen_sha256=x['frozen_sha256'],admission=n+'/'+attempt+'/admission.json',handoff=n+'/'+attempt+'/handoff.json',
                  resources=n+'/'+attempt+'/resources.json',ISA_proof=n+'/isa-proof.json') for n,x in nodes.items()},
    cpu_validation_sha256=sha(data/'cpu-validation.json'),CPU_reference_directory='external-data/work/tp-decode-v1-cpu/refs',
    large_build_artifacts=str(raw),physical_campaign1='OPEN',physical_campaign2='OPEN',GPU_executions=0,GPU_requests=0,
    next_action='ROOT revisa fuentes/recursos/ownership/admisión/presupuesto; solo ROOT aporta lease real y recibos físicos'))
r=load(data/'node1'/attempt/'resources.json');rows=[]
for row in range(1,9):
    values=[next(x['vgpr_count'] for x in r['resources'] if x['variant']==v and x['rows']==row) for v in ('A','B','C')]
    rows.append(f"| {row} | "+' | '.join(map(str,values))+' |')
binary=nodes['node1']['build']['binary_sha256'];elf=r['new_whole_ELFs'][0]
report=f'''# Candidato privado decode TP v1: compilado y congelado solo en CPU

2026-10-21. Rama `performance/tp-decode-v1`, base exacta `205ad21`.
Implementación privada preparada en ambos nodos; **ninguna ejecución, descubrimiento,
solicitud ni lease GPU**. V16 sigue siendo el default. Sin agentes, push, merge,
descargas de modelos ni escrituras en el worktree/salida del trabajador de datos.
Solo fuentes nuevas en `experiments/tp_decode`, herramientas e informes.

El [handoff](tp-decode-private-data/handoff.json) identifica las dos congelaciones.
Los datos nativos, grafos, atributos efectivos y campañas físicas siguen **OPEN**.
No hay resultados de rendimiento TP ni recomendación de promoción.

Revisión host-r2: ROOT detectó UB al leer enums C inválidos. La congelación inicial
`sealed-cpu` permanece intacta y no es admisible; su payload antiguo tiene STOP
por el hallazgo CPU, sin ejecución física. El candidato vigente está en
`tp-decode-v1/host-r2`: lee numerics/dtype por memcpy a uint32, conserva layout,
rechaza Variant desconocido antes de encolar y añade el negativo nativo de cero nodos.
El CPU test real pasó ASan+UBSan con abort al primer error en ambos nodos, también
con representaciones UINT32_MAX de numerics y ambos dtype. Las30 funciones kernel
y dos sentinelas C++ no lanzables conservan todos sus bytes FUN frente a la primera
congelación. El ELF candidato completo cambia en tablas de símbolos/strings/hash
dinámicos; no se afirma identidad de ese ELF entre revisiones host.
[Prueba y hallazgos preservados](tp-decode-private-data/host-r2-proof.json),
[sanitizer node1](tp-decode-private-data/node1/host-r2/sanitized-cpu.log),
[node2](tp-decode-private-data/node2/host-r2/sanitized-cpu.log).
La nota temporal `ROOT_REVIEW_REQUIRED.json` se reconoce y se deja sin seguimiento,
sin modificarla, como pidió su propietario ROOT.

## Identidad y controles

Archivo real V16, separado del candidato:
`8eb47b095b6ffdc3444e61da5d52dc2cf922d87da8aaec3ef4b4db8c7fed1db8`.
Consumidores internos recompilados con TEN/56 B, flags en48/49, C ABI1.
No se ha usado el archivo K3 privado de nueve campos.

El control canónico llama al `baseline_core` real del archivo con
`(w,A,P,R,R,4,stream)`, entre dos transformaciones propias con stride.
El control antiguo llama al `launch_linear` V16 completo para las formas TP
contiguas reales. Ambos consumen el mismo X F32 original, packed/suh/svh/bias;
sin modificar A8 ni excluir staging necesario en una comparación estrided.
Esta campaña solo cronometra formas contiguas; los strides son una puerta funcional.

El enlace conserva **11 ELF completos antiguos byteidénticos**, con433 kernels.
Añade un ELF de{elf['bytes']} B, SHA256 `{elf['sha256']}`, con30 funciones:
**26 cuerpos principales +2 diagnósticos D +2 transformaciones**. Son463 funciones
GPU enlazadas en12 imágenes. El inventario no es27: las dos especializaciones
`Diagnostic=true` de D se cuentan explícitamente, aparte de los26 cuerpos.
No se instancia una copia por geometría, capa, rank, stride o epílogo.
La tabla ELF tiene además `__cxa_pure_virtual` y `__cxa_deleted_virtual`, dos
sentinelas de enlace de dispositivo sin metadata de kernel ni entrada lanzable;
por eso hay32 STT_FUNC en la imagen nueva pero30 funciones consultables/lanzables.

Binario idéntico en los dos nodos,3493296 B antes de la última adición de log de hardware;
el tamaño final exacto queda en el manifest de artefactos siguiente y el hash es
`{binary}`. El hash, no ese tamaño histórico, es la identidad vinculante.
[Origen del enlace](tp-decode-private-data/node1/{attempt}/link-audit.json),
[recursos e imágenes](tp-decode-private-data/node1/{attempt}/resources.json).

## Matriz y matemática congeladas

A/B/C cubren R1..8, q_b1280→16384 y o_b4096→5120, ambos ranks, K5/CB2/S4.
A usa cargas directas R1..4 y cache LDS chunk8 R5..8. B mantiene dos tiles RAW;
C transforma H completos y recorta el intervalo K16 original.
D tiene solo R1/G4 para ambos TP y R6/G2 para o_b. No hay prefetch en D.
El whitelist del host rechaza otros dominios, enum o S antes de encolar.

Las fronteras son `floor(T*s/4)`: q_b0/20/40/60/80; o_b0/64/128/192/256.
La cache cubre `floor(b/8)..ceil(e/8)` y el dot solo `[b,e)`.
Se conserva RN32 materializada de X*suh, RN16, H128 F32 materializada y RN16;
k/p/lane, fdot2 con clamp=false y shuffles2→1; P por split lógico y suma s
ascendente desde+0. Alpha/beta/bias tienen operaciones separadas,
`-ffp-contract=off`, sin atajo alpha0 y sin lectura de Yold para beta±0.
Se reutilizan únicamente H128/hash propios de esta base, con extensión propia
del lector RAW. No se ha copiado código de terceros.

B evita el patrón condicional de promoción que produjo un wait prematuro en
la primera versión. Su bucle uniforme desenrolla dos pasos: cargar next,
dot current, reutilizar sus RAW para following, dot next y promover following.
El tail se guarda antes de formar la dirección; wrap solo dentro de40 words.
Las barreras vacías de registros hacen depender la consumición RAW del acumulador
anterior; las cargas C++ siguen visibles a LLVM, sin VMEM async oculto.
En los ocho ELF B se observan cargas RAW, `vmcnt(3)` y hash/dot del actual antes
del wait completo. Esto demuestra **orden estático**, no ciclos de latencia ocultos.
[ISA node1](tp-decode-private-data/node1/isa-proof.json),
[ISA node2](tp-decode-private-data/node2/isa-proof.json).

D conserva P[S,R,N128] en LDS y suma s0..3 al final, sin presumas por grupo.
El mismo template con bool diagnóstico añade exportaciones A/P disjuntas:
A solo desde blockIdx.x0 y solo la intersección K16 propia del split;
P por tile N128/s/r. Los dumps y las funciones/FUN son distintos y están
contabilizados. El orden/instrucciones de planificación pueden diferir por
esas stores; no se afirma igualdad de FUN. La equivalencia matemática procede
del cuerpo compartido y su gate nativo obligatorio A/P/Y, todavía OPEN.
Producción D deja A/P globales en poison: se comprueban como no usados y
**no se presentan como P interno observado**. Se valida Y de todos los clones
y A/P diagnósticos completos antes/después de cada sweep y en LAST.

## Evidencia CPU real

El trabajador de datos cerró validación completa de14 fixtures: los12 densos
de L0/21/39 y los dos heads; siete corrupciones rechazadas, incluido LAST Y,
packed, scales, eje TP, alineación y marker, con el gate hash desactivado para
comprobar semántica. Tiempo537,382 s; pico CPU520544256 B, medidos por su validador.
Su recibo y SHA exactos se conservan en
[cpu-validation.json](tp-decode-private-data/cpu-validation.json).
No se modificó su generador, worktree, manifests ni salida.

La herramienta propia `tools/tp_decode_refs.py` validó en CPU local los
251658240 valores Q de los12 densos mediante lector circular y LUT independiente
de65536 estados, la semilla/procedencia/hash y A completa con las dos RN16.
Generó P FP64 y suma de productos absolutos por S4 a partir de Q real completa,
con epílogo F32 ordenado de esos dots. **No predice P nativo ni usa P del candidato
como expected.** La referencia algebraica sin redondear sigue retenida por hash
en los fixtures originales y tiene otro contrato.

Los derivados compactos están en `external-data/work/tp-decode-v1-cpu/refs`;
se copiaron read-only bajo el scratch nuevo de ambos nodos y se verificaron todos
sus SHA256 allí. Solo los cuatro L0 entran en rendimiento. Los ocho L21/39 reciben
el mismo QA/matriz sin tuning. Head y o_b N-half siguen en segunda fase.

En ambos nodos se ejecutaron únicamente binarios CPU sin dependencia HIP/HSA:
C11 layout/align8; C++17 descriptor312 B/offsets; overflow de producto/suma/address,
extensión exacta y un float corta, R0/9, stride/enum/S/formato, alias por envolvente,
adyacencia, huecos/canarios, primer error y LAST con capacidad declarada suficiente
pero extensión propia corta. También ring por lane/estado y hash carry32 completo.
La fábrica CPU es un **mock con asignaciones propias**, no una prueba de extensión HIP.

Los16 wrappers reales con stand-ins CPU reenvían el error713 fuera de Submit y
bloquean alloc/copy/sync/query/event dentro. Los wrappers ordinarios C/C++ bloquean
heap sin definir `new/delete` globales. El contraejemplo DSO prueba que sus llamadas
internas quedan fuera de `--wrap` y siguen asignando/liberando correctamente;
no se atribuye instrumentación global del driver.
[Pruebas node1](tp-decode-private-data/node1/{attempt}/hooks_cpu.log),
[node2](tp-decode-private-data/node2/{attempt}/contract_cpu.log).
La prueba Python añade RN32/RN16 distinguibles, cancelación/subnormal, máscaras
continuas RAW con máximo dos tiles vivos, veto horario y corrupción LAST del freeze.
Sus tiempos sintéticos son exclusivamente datos de prueba del agregador,
nunca evidencia de benchmark. [Log](tp-decode-private-data/proof-cpu.log).

Fallos preservados: primera compilación `initial` en ambos nodos por retornos HIP
nodiscard de destructores, corregidos; auditoría `steady-raw` por confundir líneas
de relocación con definiciones de símbolos, corregida. Las versiones B anteriores
se retienen como evidencia del wait prematuro, no como candidato admitido.
Los ensayos locales iniciales detectaron count14 frente a13 secciones STGPU,
shadowing de módulos Python y un contraejemplo de cancelación con exponente24
en vez de25; están corregidos y no se les atribuye PASS.

## Recursos y compilación, sin ejecución HIP

VGPR emitidos, idénticos en ambos nodos:

| R | A | B | C |
|---|---:|---:|---:|
{chr(10).join(rows)}

D producción R1/R6:29/59 VGPR; diagnóstico31/60. A/B R6 cumplen≤64;
C/D cumplen≤96. Todas las30 funciones wave32, private0, spill SGPR/VGPR0;
maxThreads128 para A/B/C/transforms y1024/512 para D R1/R6.
No se retira ninguna variante por los recursos ELF actuales. SGPR y demás campos
completos están en los JSON; no se calcula ocupación a partir de20 WGP/40 CU ni
se afirma ocupación medida.

LDS estática emitida0; LDS dinámica solicitada, **calculada, no consultada aún**:
A/B R6=1536 B; C R8=6144/16384 B para q_b/o_b;
D q_bR1=5120, o_bR1=10240, o_bR6=36864 B.
La preparación nativa consultará una vez cada función/contexto, imprimirá
atributos reales maxThreads/dynamic/static/local/regs y comprobará la capacidad
real antes del primer Submit. No hay SetAttribute ni consultas de recursos
post-prepare en Submit. Su puerta efectiva sigue OPEN.

ROCm HIP7.16.26385, clang24.0.0git d6f6cb691863e9258034ab48c1dfce21689d5606;
gfx1151, O2, sin fast math, ffp-contract off. Cada build remoto se ejecutó con
nice10, un trabajo por nodo, TMP/XDG/HIP/COMGR privados bajo el scratch nuevo.
Comandos completos, errores y `/usr/bin/time -v` se conservan por intento.
El tiempo build+inspección final fue{nodes['node1']['build']['build_seconds']:.3f}/
{nodes['node2']['build']['build_seconds']:.3f} s; es tiempo CPU de preparación.
No se ejecutó `tp_native`, ni siquiera con --help; tampoco consultas/discovery GPU.

## Contrato privado y gates nativos preparados

Prepared retiene las asignaciones hipMalloc, streams, pesos y recursos hasta LAST.
Graph retiene Prepared y destruye grafos antes de liberar sus owners; el harness
sobrescribe el descriptor host con otro owner válido después de capturar y
comprueba que replay no retargetea esos otros buffers. Dos streams se encolan
antes de esperar cualquiera, también con dos replays. Esto es implementación
privada pendiente de gate nativo, no garantía nueva para punteros públicos prestados.

La validación completa precede cualquier enqueue. Las capacidades declaradas se
comprueban contra extents inmutables de la fábrica propia, no por una nueva query
hipMemGetAddressRange de memoria ajena. La extensión es
`4*((R-1)*stride+width)`; se admite R1 con stride enorme. Los negativos capturan
cero nodos y comprueban buffers intactos, incluido LAST de extensión propia.
Un error después de encolar no promete rollback.

El QA futuro cubre cada una de8 filas en cada posición/R mediante ventanas
cíclicas, permutación/reverso, vecinos cambiados a otras filas referenciadas,
particiones1+7/3+5/6+2, rechazo/reintento, rank1 y strides distintos.
Comprueba A/P/Y completos por bytes frente al canónico y P/Y contra CPU por fila
y split: NRMS≤3e-5 y maxabs≤5e-4*max(1,maxabs(ref)), más gamma_2K*sumabs.
Incluye bias sí/no, alpha0/1/negativo, beta+0/-0/1/negativo y YoldNaN con beta0.
V16 antiguo registra igualdad A/P/Y observada y tolerancias por separado;
no se ha observado ni supuesto aún igualdad nativa antigua.

## Dos campañas futuras, sin autoarranque

134 puntos de comparación de los cuatro primarios: canónico/A/B/C todos R,
D solo elegibles. Cada punto tiene21 packed disjuntos de13107200 B:
275251200 B residentes y recorridos, ≥256 MiB. Once sweeps alternan cronológicamente
V16/candidato por punto; cada muestra es un operador lógico R real, con evento
y wall que incluye el sync final honesto. Preparar/asignar/subir queda fuera.
No se resta timing global ni se divide un tiempo de pool entre21.
61908 muestras previstas, sin contar QA/dumps. Min/mediana/p90, todos los sweeps,
pérdidas y colas se conservan en crudo y en `summarize.py`.

A/P/Y y canarios completos de cada clon se comprueban antes y después de cada
sweep y LAST, además de packed/scales/bias completos. D añade su diagnóstico fuera
de eventos. Este volumen de comprobación no se reducirá para ajustar el presupuesto.
No se afirma que la campaña quepa por una medida aún inexistente.

Estimaciones de memoria **sin medir**: pico GPU propio<450 MiB y host propio<300 MiB,
un pool/geometría cada vez, antes de sobrecargas del driver. Límite duro de
asignaciones GPU propias1 GiB y puerta RSS host2 GiB. RSS/memoria reales y duración
de QA+bench siguen OPEN; ROOT debe revisar presupuesto. Inner850 s, outer900 s,
sin repetición automática. Primer fallo numérico, recursos u ownership crea STOP
y preserva raw logs. No se rebajan gates ni se cambia R/G/S para reintentar.

Ruta vigente en ambos nodos: `~/gpu-site/scratch-trellis/tp-decode-v1/host-r2`.
Antes de esperar lease, ROOT ejecutará únicamente el chequeo CPU:

```sh
cd ~/gpu-site/scratch-trellis/tp-decode-v1/host-r2
python3 -I -S -B experiments/tp_decode/admission.py --prequeue
```

Solo dentro de un lease real posterior de ROOT, con recibo externo de900 s,
label nuevo, campaña1/2 y revisiones source/resources/ownership/admission/budget
PASS aportadas por ROOT:

```sh
ROOT_GPU_LEASE=1 bash experiments/tp_decode/root_payload.sh --inside-root-lease ROOT_RECEIPT.json
```

El payload nunca solicita ni espera GPU. Revalida mantenimiento y veto local
21:00–23:30 después de la espera externa, fuentes/freeze/runtime antes/después,
y crea un claim exclusivo por campaña. ROOT conserva los recibos reales del outer;
no se ha fabricado ninguno. Ambas campañas siguen OPEN. Una recomendación exige
>2% en evento y host en ambas campañas y todos los gates exactos; hoy no hay ninguna.

## Reproducción y congelaciones

CPU local:

```powershell
python -B tools/tp_decode_refs.py --source external-data/work/tp-real-fixtures-cpu --out external-data/work/tp-decode-v1-cpu/refs
python -B experiments/tp_decode/proof_cpu.py
```

La primera herramienta rehúsa sobrescribir la salida: el comando anterior ya
se ejecutó. Los logs SSH contienen los comandos reales de stage/build/fixtures/freeze.
Entrada remota de compilación final, ya ejecutada por `tp_decode_stage.py`:

```sh
nice -n 10 python3 -B experiments/tp_decode/build.py host-r2
```

Los builds también rehúsan repetir un nombre de intento. Freeze node1:
`{nodes['node1']['frozen_sha256']}`; node2:
`{nodes['node2']['frozen_sha256']}`. Difieren por logs/metadatos medidos por nodo;
fuentes y binario coinciden. Se verificó SHA256 completo en ambos después de cerrar.
Los ELF/ISA/maps/binarios y source tar completos permanecen fuera de Git en el
scratch nuevo y `external-data/work/tp-decode-v1-build`; Git solo contiene fuentes,
manifests, pruebas y logs pequeños, sin pesos ni binarios.
'''
# Derive final binary size instead of carrying a historical literal into the report.
size=(raw/'node1'/attempt/'tp_native').stat().st_size
report=report.replace('Binario idéntico en los dos nodos,3493296 B antes de la última adición de log de hardware;\nel tamaño final exacto queda en el manifest de artefactos siguiente y el hash es\n',f'Binario idéntico en los dos nodos,{size} B, SHA256\n').replace('El hash, no ese tamaño histórico, es la identidad vinculante.','')
(repo/'reports/tp-decode-private-candidate.md').write_text('\n'.join(line.rstrip() for line in report.splitlines())+'\n',encoding='utf-8')
print('PASS Spanish CPU-only report and open ROOT handoff')
