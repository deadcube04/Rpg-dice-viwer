import { useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import { Action, Notice } from '@/components/ui'
import { imageUrl } from '@/lib/api'
import type { Frame, Instance } from '@/lib/contracts'

type Point = { x: number; y: number }
type Drag = { start: Point; box: Instance['bbox_xyxy']; id: string; mode: 'move' | 'resize' | 'create' } | { start: Point; mode: 'pan'; origin: Point }
export function ImageEditor({ frame, instances, selected, onSelect, onChange, disabled }: {
  frame: Frame; instances: Instance[]; selected: string | null; onSelect: (id: string) => void;
  onChange: (instances: Instance[]) => void; disabled: boolean
}) {
  const svg = useRef<SVGSVGElement>(null)
  const container = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(1)
  const [origin, setOrigin] = useState<Point>({ x: 0, y: 0 })
  const [tool, setTool] = useState<'select' | 'create' | 'pan'>('select')
  const [overlay, setOverlay] = useState(true)
  const [drag, setDrag] = useState<Drag | null>(null)
  const [error, setError] = useState<string | null>(null)
  function point(event: PointerEvent<SVGSVGElement>): Point {
    const matrix = svg.current?.getScreenCTM()
    if (!matrix) return { x: 0, y: 0 }
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return { x: Math.max(0, Math.min(frame.width, p.x)), y: Math.max(0, Math.min(frame.height, p.y)) }
  }
  function down(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0 || disabled) return
    const start = point(event)
    event.currentTarget.setPointerCapture(event.pointerId)
    if (tool === 'pan') { setDrag({ start: { x: event.clientX, y: event.clientY }, mode: 'pan', origin }); return }
    if (tool === 'create') {
      const id = crypto.randomUUID()
      const box: Instance['bbox_xyxy'] = [start.x, start.y, Math.min(start.x + 1, frame.width), Math.min(start.y + 1, frame.height)]
      onChange([...instances, { instance_id: id, bbox_xyxy: box, box_kind: 'body', polygon: null,
        die_type: null, face_token: null, face_value: null, type_known: false, value_known: false,
        detection_known: true, result_ambiguous: false, occluded: false, truncated: false, modality: 'unknown', convention: '' }])
      onSelect(id); setDrag({ start, box, id, mode: 'create' }); return
    }
    const target = event.target
    if (!(target instanceof SVGElement)) return
    const id = target.closest('[data-instance]')?.getAttribute('data-instance')
    const item = instances.find(candidate => candidate.instance_id === id)
    if (item) { onSelect(item.instance_id); setDrag({ start, box: item.bbox_xyxy, id: item.instance_id, mode: target.getAttribute('data-handle') ? 'resize' : 'move' }) }
  }
  function move(event: PointerEvent<SVGSVGElement>) {
    if (!drag || disabled) return
    if (drag.mode === 'pan') {
      const scale = svg.current?.getScreenCTM()?.a ?? 1
      setOrigin({ x: drag.origin.x - (event.clientX - drag.start.x) / scale,
        y: drag.origin.y - (event.clientY - drag.start.y) / scale })
      return
    }
    const next = point(event)
    const dx = next.x - drag.start.x, dy = next.y - drag.start.y
    const [x1, y1, x2, y2] = drag.box
    let box: Instance['bbox_xyxy']
    if (drag.mode === 'create') box = [Math.min(drag.start.x, next.x), Math.min(drag.start.y, next.y), Math.max(drag.start.x, next.x), Math.max(drag.start.y, next.y)]
    else if (drag.mode === 'resize') box = [x1, y1, Math.max(x1 + 1, Math.min(frame.width, x2 + dx)), Math.max(y1 + 1, Math.min(frame.height, y2 + dy))]
    else {
      const x = Math.max(0, Math.min(frame.width - (x2 - x1), x1 + dx))
      const y = Math.max(0, Math.min(frame.height - (y2 - y1), y1 + dy))
      box = [x, y, x + x2 - x1, y + y2 - y1]
    }
    onChange(instances.map(item => item.instance_id === drag.id ? { ...item, bbox_xyxy: box, polygon: null } : item))
  }
  function finish() {
    if (disabled) { setDrag(null); return }
    if (drag && 'id' in drag) onChange(instances.filter(item => item.instance_id !== drag.id || (item.bbox_xyxy[2] - item.bbox_xyxy[0] >= 1 && item.bbox_xyxy[3] - item.bbox_xyxy[1] >= 1)))
    setDrag(null)
  }
  const stroke = Math.max(frame.width / 400, 1) / zoom
  return <div ref={container} className="min-w-0 bg-slate-100 p-3">
    <div className="mb-3 flex flex-wrap gap-2" aria-label="Ferramentas da imagem">
      <Action primary={tool === 'select'} onClick={() => setTool('select')}>Selecionar / mover</Action>
      <Action primary={tool === 'create'} onClick={() => setTool('create')} disabled={disabled}>Criar caixa</Action>
      <Action primary={tool === 'pan'} onClick={() => setTool('pan')}>Deslocar</Action>
      <Action onClick={() => setZoom(Math.min(8, zoom * 1.5))}>Zoom +</Action><Action onClick={() => setZoom(Math.max(1, zoom / 1.5))}>Zoom −</Action>
      <Action onClick={() => { setZoom(1); setOrigin({ x: 0, y: 0 }) }}>Ajustar</Action>
      <Action onClick={() => setOverlay(!overlay)}>{overlay ? 'Ocultar caixas' : 'Mostrar caixas'}</Action>
      <Action onClick={() => { void (document.fullscreenElement ? document.exitFullscreen() : container.current?.requestFullscreen())?.catch(() => setError('Tela cheia indisponível neste navegador')) }}>Tela cheia</Action>
    </div>
    {error && <Notice error>{error}</Notice>}
    {frame.width > 0 ? <svg ref={svg} className="h-[55vh] min-h-64 w-full touch-none bg-slate-900 lg:h-[65vh]" viewBox={`${origin.x} ${origin.y} ${frame.width / zoom} ${frame.height / zoom}`}
      role="img" aria-label="Imagem com caixas editáveis; use a lista e as coordenadas para edição por teclado"
      onPointerDown={down} onPointerMove={move} onPointerUp={finish} onPointerCancel={finish}>
      <image href={imageUrl(frame.image_id)} width={frame.width} height={frame.height} />
      {overlay && instances.map((item, index) => {
        const [x1, y1, x2, y2] = item.bbox_xyxy
        const chosen = item.instance_id === selected
        return <g key={item.instance_id} data-instance={item.instance_id}>
          <rect x={x1} y={y1} width={x2-x1} height={y2-y1} fill={chosen ? '#0f766e33' : '#00000000'} stroke={chosen ? '#5eead4' : item.box_kind === 'numeral' ? '#fbbf24' : '#ffffff'} strokeWidth={stroke} />
          {item.polygon && <polygon points={item.polygon.map(p => p.join(',')).join(' ')} fill="none" stroke="#fbbf24" strokeWidth={stroke} />}
          <text x={x1 + stroke * 2} y={Math.max(stroke * 10, y1 - stroke * 2)} fill="white" stroke="#172a2a" paintOrder="stroke" strokeWidth={stroke / 2} fontSize={stroke * 8}>{index + 1} · {item.die_type ?? '?'} · {item.face_token ?? '?'}</text>
          {chosen && <rect data-handle="resize" x={x2 - stroke * 5} y={y2 - stroke * 5} width={stroke * 10} height={stroke * 10} fill="#5eead4" />}
        </g>
      })}
    </svg> : <Notice error>Imagem ilegível. Registre a exclusão com motivo.</Notice>}
    <p className="mt-2 text-xs text-slate-600">Zoom {zoom.toFixed(1)}× · {frame.width} × {frame.height} px após EXIF · Arraste o canto inferior direito para redimensionar.</p>
  </div>
}
