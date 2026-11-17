import { Button, Checkbox, ConfigProvider, Input, Radio, Select, Progress as AntProgress } from 'antd'
import type { ReactNode } from 'react'

export function AuditTheme({ children }: { children: ReactNode }) {
  return <ConfigProvider theme={{ token: { colorPrimary: '#0f766e', borderRadius: 4, fontFamily: 'system-ui, sans-serif', colorText: '#172a2a' } }}>{children}</ConfigProvider>
}
export function Action({ children, onClick, disabled = false, primary = false, busy = false, danger = false, label, pressed }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; primary?: boolean; busy?: boolean; danger?: boolean; label?: string; pressed?: boolean
}) {
  return <Button type={primary ? 'primary' : 'default'} onClick={onClick} disabled={disabled} loading={busy} danger={danger} aria-label={label} aria-pressed={pressed}>{children}</Button>
}
export function Field({ label, value, onChange, multiline = false, disabled = false }: {
  label: string; value: string; onChange: (value: string) => void; multiline?: boolean; disabled?: boolean
}) {
  return <label className="flex min-w-0 flex-col gap-1 text-xs font-medium text-slate-600"><span>{label}</span>
    {multiline ? <Input.TextArea aria-label={label} value={value} onChange={e => onChange(e.target.value)} rows={2} disabled={disabled} />
      : <Input aria-label={label} value={value} onChange={e => onChange(e.target.value)} disabled={disabled} />}</label>
}
export function Choice({ label, value, options, onChange, disabled = false }: {
  label: string; value: string; options: { value: string; label: string }[]; onChange: (value: string) => void; disabled?: boolean
}) {
  return <label className="flex min-w-0 flex-col gap-1 text-xs font-medium text-slate-600"><span>{label}</span>
    <Select<string> aria-label={label} className="w-full" value={value} options={options} onChange={onChange} disabled={disabled}
      showSearch={{ optionFilterProp: 'label' }} notFoundContent="Nenhuma opção encontrada" /></label>
}
export function RadioChoices({ label, name, value, options, onChange, disabled = false }: {
  label: string; name: string; value: string; options: { value: string; label: string }[]; onChange: (value: string) => void; disabled?: boolean
}) {
  return <fieldset className="min-w-0"><legend className="mb-2 text-xs font-medium text-slate-600">{label}</legend>
    <Radio.Group aria-label={label} name={name} value={value} disabled={disabled} className="flex w-full flex-col gap-2"
      onChange={event => { const next: unknown = event.target.value; if (typeof next === 'string') onChange(next) }}>
      {options.map(option => <Radio key={option.value} value={option.value}>{option.label}</Radio>)}
    </Radio.Group></fieldset>
}
export function Flag({ label, checked, onChange, disabled = false }: { label: string; checked: boolean; onChange: (value: boolean) => void; disabled?: boolean }) {
  return <Checkbox checked={checked} onChange={e => onChange(e.target.checked)} disabled={disabled}>{label}</Checkbox>
}
export function Meter({ complete, total }: { complete: number; total: number }) {
  const percent = total > 0 ? Math.min(100, Math.max(0, complete / total * 100)) : 0
  return <AntProgress percent={percent} size="small" aria-label={`${complete} de ${total} imagens revisadas`} aria-valuenow={percent}
    format={() => `${percent.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`} />
}
export function Notice({ children, error = false }: { children: ReactNode; error?: boolean }) {
  return <p role={error ? 'alert' : 'status'} className={`rounded border p-3 text-sm ${error ? 'border-red-300 bg-red-50 text-red-900' : 'border-teal-200 bg-teal-50 text-teal-950'}`}>{children}</p>
}
