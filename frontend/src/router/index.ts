import { createRouter, createWebHistory } from "vue-router"

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      component: () => import("@/layout/AppLayout.vue"),
      redirect: "/dashboard",
      children: [
        {
          path: "dashboard",
          name: "Dashboard",
          component: () => import("@/views/dashboard/Index.vue"),
          meta: { title: "系统看板", icon: "dashboard" }
        },
        {
          path: "data",
          name: "Data",
          component: () => import("@/views/data/Index.vue"),
          meta: { title: "数据管理", icon: "data" }
        },
        {
          path: "factors",
          name: "Factors",
          component: () => import("@/views/factors/Index.vue"),
          meta: { title: "因子管理", icon: "factor" }
        },
        {
          path: "strategies",
          name: "Strategies",
          component: () => import("@/views/strategies/Index.vue"),
          meta: { title: "策略管理", icon: "strategy" }
        },
        {
          path: "backtest",
          name: "Backtest",
          component: () => import("@/views/backtest/Index.vue"),
          meta: { title: "回测管理", icon: "backtest" }
        },
        {
          path: "market",
          name: "Market",
          component: () => import("@/views/market/Index.vue"),
          meta: { title: "市场行情", icon: "market" }
        }
      ]
    }
  ]
})

export default router
