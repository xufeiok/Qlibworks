<template>
  <div>
    <n-h2>策略管理</n-h2>
    <n-p>ML 模型策略 | Optuna 优化 | 策略组合</n-p>
    <n-grid :x-gap="12" :y-gap="12" :cols="4">
      <n-grid-item v-for="i in 4" :key="i">
        <n-card :title="'策略 ' + i" hoverable>
          <n-skeleton text :repeat="3" />
        </n-card>
      </n-grid-item>
    </n-grid>
    <n-empty description="正在开发中..." style="margin-top:24px" />
  </div>
</template>
<script setup lang="ts">
</script>
