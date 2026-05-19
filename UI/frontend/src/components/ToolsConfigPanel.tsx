import { useEffect, useState } from 'react'
import { getToolsRegistry } from '../api/client'
import type { ToolCategory } from '../types'

interface Props {
  onChange: (selectedTools: string[]) => void
}

export default function ToolsConfigPanel({ onChange }: Props) {
  const [categories, setCategories] = useState<ToolCategory[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [initialised, setInitialised] = useState(false)

  useEffect(() => {
    getToolsRegistry()
      .then(registry => {
        setCategories(registry.categories)
        // Default: all tools selected
        const defaults = new Set<string>()
        for (const cat of registry.categories) {
          for (const tool of cat.tools) {
            defaults.add(tool.id)
          }
        }
        setSelected(defaults)
        setInitialised(true)
      })
      .catch(() => {})
  }, [])

  // Notify parent when selection changes (skip initial render)
  useEffect(() => {
    if (initialised) {
      onChange(Array.from(selected))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, initialised])

  function toggleTool(id: string) {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function toggleCategory(catId: string) {
    setCollapsed(prev => {
      const next = new Set(prev)
      if (next.has(catId)) next.delete(catId)
      else next.add(catId)
      return next
    })
  }

  function toggleAllInCategory(cat: ToolCategory) {
    const ids = cat.tools.map(t => t.id)
    if (ids.length === 0) return
    const allSelected = ids.every(id => selected.has(id))
    setSelected(prev => {
      const next = new Set(prev)
      for (const id of ids) {
        if (allSelected) next.delete(id)
        else next.add(id)
      }
      return next
    })
  }

  if (categories.length === 0) return null

  return (
    <div className="space-y-1">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
        Tools
      </h3>
      {categories.map(cat => {
        const isCollapsed = collapsed.has(cat.id)
        const totalCount = cat.tools.length
        const selectedCount = cat.tools.filter(t => selected.has(t.id)).length

        return (
          <div key={cat.id} className="border border-gray-200 rounded">
            <button
              type="button"
              onClick={() => toggleCategory(cat.id)}
              className="flex w-full items-center justify-between px-2 py-1.5 text-left text-xs font-medium text-gray-700 hover:bg-gray-50"
            >
              <span className="flex items-center gap-1">
                <span className={`transition-transform ${isCollapsed ? '' : 'rotate-90'}`}>
                  &#9654;
                </span>
                {cat.name}
              </span>
              {totalCount > 0 && (
                <span className="text-[10px] text-gray-400">
                  {selectedCount}/{totalCount}
                </span>
              )}
            </button>

            {!isCollapsed && (
              <div className="border-t border-gray-100 px-2 py-1 space-y-0.5">
                {cat.tools.length === 0 ? (
                  <p className="text-[10px] italic text-gray-400 py-0.5">No tools available</p>
                ) : (
                  <>
                    {totalCount > 1 && (
                      <button
                        type="button"
                        onClick={() => toggleAllInCategory(cat)}
                        className="text-[10px] text-blue-500 hover:text-blue-700 mb-0.5"
                      >
                        {cat.tools.every(t => selected.has(t.id))
                          ? 'Deselect all'
                          : 'Select all'}
                      </button>
                    )}
                    {cat.tools.map(tool => (
                      <label
                        key={tool.id}
                        className="flex items-center gap-1.5 text-xs py-0.5 text-gray-700 cursor-pointer"
                        title={tool.description || ''}
                      >
                        <input
                          type="checkbox"
                          className="h-3 w-3 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          checked={selected.has(tool.id)}
                          onChange={() => toggleTool(tool.id)}
                        />
                        <span>{tool.name}</span>
                      </label>
                    ))}
                  </>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
