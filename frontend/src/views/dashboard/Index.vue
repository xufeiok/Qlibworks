<template>
  <div>
    <n-h2>系统看板</n-h2>
    <n-p style="color:#8b949e;margin-bottom:16px">数据概览 | 系统状态</n-p>
    <n-grid :x-gap="12" :y-gap="12" :cols="4" style="margin-bottom:16px">
      <n-grid-item v-for="c in cards" :key="c.l">
        <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
          <n-statistic :label="c.l" :value="c.v"><template #suffix><n-text depth="3" style="font-size:11px">{{ c.u }}</n-text></template></n-statistic>
        </n-card>
      </n-grid-item>
    </n-grid>
    <n-grid :x-gap="12" :y-gap="12" :cols="2">
      <n-grid-item>
        <n-card title="数据概览" :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
          <n-empty description="暂无数据统计" />
        </n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card title="系统状态" :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
          <n-thing><template #description><n-p>运行时间: <n-tag size="small" type="success">在线</n-tag></n-p><n-p>API 连接: <n-tag size="small" type="warning">未连接</n-tag></n-p><n-p>WebSocket: <n-tag size="small" type="default">未连接</n-tag></n-p></template></n-thing>
        </n-card>
      </n-grid-item>
    </n-grid>
  </div>
</template>
<script setup lang="ts">
const cards = [
  { l: "股票总数", v: "5,124", u: "A股 + 港股" },
  { l: "K线总数", v: "48.2M", u: "1990 ~ 至今" },
  { l: "因子数量", v: "1,247", u: "YAML 配置" },
  { l: "量化策略", v: "3", u: "近30天" },
]
</script>
