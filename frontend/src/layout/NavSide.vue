<template>
  <n-layout-sider bordered :width="200" :collapsed-width="64" show-trigger="bar" :style="{ background: '#161b22' }">
    <n-menu :value="activeKey" :options="menuOptions" @update:value="handleMenuSelect" :style="{ paddingTop: '8px' }" />
  </n-layout-sider>
</template>
<script setup lang="ts">
import { computed } from "vue"
import { useRouter, useRoute } from "vue-router"
const router = useRouter()
const route = useRoute()
const menuOptions = [
  { label: "📊 首页看板", key: "/dashboard" },
  { label: "🗄️ 数据管理", key: "/data" },
  { label: "🧮 因子管理", key: "/factors" },
  { label: "🤖 策略管理", key: "/strategies" },
  { label: "🎯 回测管理", key: "/backtest" },
  { label: "📈 市场行情", key: "/market" },
]
const activeKey = computed(() => { for (const m of menuOptions) { if (route.path.startsWith(m.key)) return m.key } return "/dashboard" })
const handleMenuSelect = (key: string) => router.push(key)
</script>