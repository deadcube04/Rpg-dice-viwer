import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRootRoute, createRoute, createRouter, Outlet, RouterProvider } from '@tanstack/react-router'
import { z } from 'zod'
import { App } from '@/App'
import { AuditTheme } from '@/components/ui'
import '@/styles.css'

const filterSchema = z.object({ source: z.string().catch(''), decision: z.string().optional(), type: z.string().catch(''),
  count: z.string().catch(''), issue: z.string().catch(''), search: z.string().catch(''),
  double_check: z.preprocess((value: unknown) => value === true || value === 'true', z.boolean()).catch(false),
  offset: z.coerce.number().int().nonnegative().catch(0) })
  .transform(filters => ({ ...filters, decision: filters.decision ?? (filters.double_check ? '' : 'pending') }))
const route = createRootRoute({ component: Outlet, validateSearch: input => filterSchema.parse(input) })
const index = createRoute({ getParentRoute: () => route, path: '/', component: App })
const router = createRouter({ routeTree: route.addChildren([index]) })
declare module '@tanstack/react-router' { interface Register { router: typeof router } }
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } })
const element = document.getElementById('root')
if (!element) throw new Error('Elemento root ausente')
ReactDOM.createRoot(element).render(<React.StrictMode><QueryClientProvider client={queryClient}><AuditTheme><RouterProvider router={router} /></AuditTheme></QueryClientProvider></React.StrictMode>)
