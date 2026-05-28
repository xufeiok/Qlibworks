<template>
  <n-layout-header bordered :style="{ height: '48px', display: 'flex', alignItems: 'center', padding: '0 16px', background: '#161b22' }">
    <div :style="{ fontWeight: 700, fontSize: '15px', color: '#39d2c0', display: 'flex', alignItems: 'center', gap: '8px' }">
      <n-icon size="20"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 16l4-8 4 4 4-6"/></svg></n-icon>
      QlibWorks
    </div>
    <n-space :style="{ marginLeft: '24px' }" item-style="display:flex">
      <router-link v-for="item in navItems" :key="item.path" :to="item.path"
        style="text-decoration:none;font-size:13px;padding:4px 10px;border-radius:4px;color:#8b949e"
        :style="isActive(item.path) ? 'color:#39d2c0;background:rgba(57,210,192,0.08)' : ''">
        <n-text>{{ item.label }}</n-text>
      </router-link>
    </n-space>
    <div style="margin-left:auto">
      <n-tag size="small" type="success" round>v0.1 dev</n-tag>
    </div>
  </n-layout-header>
</template>

<script setup lang="ts">
import { useRoute } from "vue-router"
const route = useRoute()
const navItems = [
  { path: "/dashboard", label: "首页" },
  { path: "/data", label: "数据" },
  { path: "/factors", label: "因子" },
  { path: "/strategies", label: "策略" },
  { path: "/backtest", label: "回测" },
  { path: "/market", label: "行情" },
]
const isActive = (p: string) => route.path.startsWith(p)
</script>
