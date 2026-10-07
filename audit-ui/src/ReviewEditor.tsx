import { useEffect, useState } from 'react'
import { AnnotationCanvas } from './AnnotationCanvas'
import type { Box, Die, DieType, Frame, Review } from './types'

const TYPES: DieType[] = ['d4', 'd6', 'd8', 'd10', 'd12', 'd20']
type Props = { frame: Frame; saving: boolean; onSave: (review: Review) => void }

function safeDie(type: DieType, box: Box): Die {
  return { type, value: null, box }
}

export function ReviewEditor({ frame, saving, onSave }: Props) {
  const [dice, setDice] = useState<Die[]>(frame.dice)
  const [selected, setSelected] = useState<number | null>(frame.dice.length ? 0 : null)
  const [trusted, setTrusted] = useState(frame.values_trusted)
  const [note, setNote] = useState(frame.note)
  const [message, setMessage] = useState('')

  useEffect(() => {
    setDice(frame.dice)
    setSelected(frame.dice.length ? 0 : null)
    setTrusted(frame.values_trusted)
    setNote(frame.note)
    setMessage('')
  }, [frame.id, frame.reviewed, frame.dice, frame.values_trusted, frame.note])

  function updateDie(index: number, updated: Die) {
    setDice(current => current.map((item, position) => position === index ? updated : item))
  }

  function addBox(box: Box) {
    setDice(current => [...current, safeDie('d6', box)])
    setSelected(dice.length)
    setMessage('Novo dado adicionado. Selecione o tipo e o valor.')
  }

  const die = selected === null ? null : dice[selected] ?? null
  const canTrust = dice.every(item => item.value !== null &&
    ((item.type === 'd10' && item.value === 0) || (item.value >= 1 && item.value <= Number(item.type.slice(1)))))

  return <div className="workspace">
    <div className="frame-heading">
      <div><div className="eyebrow">IMAGEM {frame.id}</div><h2>Revisar anotação</h2>
        <p>{frame.camera} <span className="dot">·</span> {frame.epoch} <span className="dot">·</span> {frame.width} × {frame.height}</p></div>
      <span className={`split-badge ${frame.split}`}>{frame.split === 'train' ? 'Treino' : frame.split === 'valid' ? 'Validação' : 'Teste reservado'}</span>
    </div>
    <div className="editor-layout">
      <AnnotationCanvas frame={frame} dice={dice} selected={selected} onSelect={setSelected} onAdd={addBox} />
      <aside className="editor-panel" aria-label="Edição dos rótulos">
        <div className="panel-section">
          <div className="section-title"><h3>Dados na imagem</h3><span>{dice.length}</span></div>
          {dice.length === 0 && <p className="muted">Nenhum dado anotado. Desenhe uma caixa na foto se encontrar um.</p>}
          <div className="die-list">{dice.map((item, index) => <button type="button" key={index}
            className={`die-row ${selected === index ? 'active' : ''}`} onClick={() => setSelected(index)}>
            <span className="die-index">{String(index + 1).padStart(2, '0')}</span>
            <strong>{item.type.toUpperCase()}</strong><span>Face {item.value ?? '?'}</span>
          </button>)}</div>
          <button type="button" className="add-button" onClick={() => addBox({ x: 0.4, y: 0.4, w: 0.2, h: 0.2 })}>
            + Adicionar dado
          </button>
        </div>
        {die && selected !== null && <div className="panel-section">
          <div className="section-title"><h3>Dado {selected + 1}</h3><button type="button" className="text-danger" onClick={() => {
            setDice(current => current.filter((_, index) => index !== selected))
            setSelected(null)
          }}>Remover</button></div>
          <div className="fields two"><label>Tipo<select value={die.type} onChange={event => {
            const type = event.target.value as DieType
            updateDie(selected, { ...die, type, value: die.value !== null && die.value <= Number(type.slice(1)) ? die.value : null })
          }}>{TYPES.map(type => <option key={type} value={type}>{type.toUpperCase()}</option>)}</select></label>
          <label>Valor<input type="number" min={die.type === 'd10' ? 0 : 1} max={Number(die.type.slice(1))}
            value={die.value ?? ''} onChange={event => updateDie(selected, { ...die, value: event.target.value === '' ? null : Number(event.target.value) })} /></label></div>
          <p className="field-note">No D10, a face 0 vira 10. D4 permanece anotado, mas não entra no treino do leitor atual.</p>
          <div className="coordinate-heading">Caixa normalizada</div>
          <div className="fields four">{(['x', 'y', 'w', 'h'] as const).map(key => <label key={key}>{key.toUpperCase()}
            <input type="number" min="0" max="1" step="0.001" value={Number(die.box[key].toFixed(4))}
              onChange={event => updateDie(selected, { ...die, box: { ...die.box, [key]: Number(event.target.value) } })} />
          </label>)}</div>
        </div>}
        <div className="panel-section review-section">
          <h3>Confiabilidade</h3>
          <label className="check-line"><input type="checkbox" checked={trusted} onChange={event => setTrusted(event.target.checked)} />
            <span>Tipos e valores legíveis e corretos</span></label>
          <p className="field-note">Desmarque para manter as caixas no detector sem usar os valores no leitor.</p>
          <label className="notes-label">Nota de revisão<textarea value={note} maxLength={1000}
            placeholder="Ex.: reflexo no D20; face ilegível" onChange={event => setNote(event.target.value)} /></label>
          {message && <p className="inline-message" role="status">{message}</p>}
          <button type="button" className="save-button" disabled={saving || (trusted && !canTrust)}
            onClick={() => onSave({ dice, values_trusted: trusted, note })}>{saving ? 'Salvando…' : 'Salvar revisão'}</button>
          {trusted && !canTrust && <p className="field-note warning">Preencha todos os valores ou desmarque a confiabilidade.</p>}
        </div>
      </aside>
    </div>
  </div>
}
