import { useRef, useState } from 'react'
import { imageUrl } from './api'
import type { Box, Die, Frame } from './types'

type Props = {
  frame: Frame
  dice: Die[]
  selected: number | null
  onSelect: (index: number) => void
  onAdd: (box: Box) => void
}

function point(event: React.PointerEvent<HTMLDivElement>, element: HTMLDivElement): { x: number; y: number } {
  const bounds = element.getBoundingClientRect()
  return {
    x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
    y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
  }
}

function between(start: { x: number; y: number }, end: { x: number; y: number }): Box {
  return { x: Math.min(start.x, end.x), y: Math.min(start.y, end.y),
    w: Math.abs(end.x - start.x), h: Math.abs(end.y - start.y) }
}

export function AnnotationCanvas({ frame, dice, selected, onSelect, onAdd }: Props) {
  const surface = useRef<HTMLDivElement>(null)
  const [start, setStart] = useState<{ x: number; y: number } | null>(null)
  const [draft, setDraft] = useState<Box | null>(null)

  return <div className="image-stage">
    <div className="image-surface" ref={surface} style={{ aspectRatio: `${frame.width} / ${frame.height}` }}
      onPointerDown={event => {
        if (event.button !== 0 || event.target !== event.currentTarget) return
        const origin = point(event, event.currentTarget)
        setStart(origin)
        setDraft({ ...origin, w: 0, h: 0 })
        event.currentTarget.setPointerCapture(event.pointerId)
      }}
      onPointerMove={event => {
        if (start && surface.current) setDraft(between(start, point(event, surface.current)))
      }}
      onPointerUp={event => {
        if (start && surface.current) {
          const box = between(start, point(event, surface.current))
          if (box.w >= 0.015 && box.h >= 0.015) onAdd(box)
        }
        setStart(null)
        setDraft(null)
      }}>
      <img src={imageUrl(frame.id)} alt={`Fotografia ${frame.id} com ${dice.length} dados anotados`} draggable={false} />
      {dice.map((die, index) => <button type="button" key={index}
        className={`annotation-box ${selected === index ? 'selected' : ''}`}
        style={{ left: `${die.box.x * 100}%`, top: `${die.box.y * 100}%`,
          width: `${die.box.w * 100}%`, height: `${die.box.h * 100}%` }}
        aria-label={`Selecionar dado ${index + 1}: ${die.type.toUpperCase()}, valor ${die.value ?? 'não informado'}`}
        onClick={() => onSelect(index)}><span>{index + 1} · {die.type.toUpperCase()} · {die.value ?? '?'}</span></button>)}
      {draft && <div className="annotation-draft" style={{ left: `${draft.x * 100}%`, top: `${draft.y * 100}%`,
        width: `${draft.w * 100}%`, height: `${draft.h * 100}%` }} />}
    </div>
    <p className="image-hint">Arraste em uma área vazia para adicionar uma caixa. Clique em uma caixa para editá-la.</p>
  </div>
}
