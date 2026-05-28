<template>
  <n-layout position="absolute" has-sider>
    <NavSide />
    <n-layout>
      <TopNav />
      <n-layout-content
        :style="{ padding: '24px', background: 'var(--body-color)', minHeight: 'calc(100vh - 48px)' }"
        content-style="max-width: 1400px; margin: 0 auto;"
      >
        <router-view />
      </n-layout-content>
    </n-layout>
  </n-layout>
</template>

<script setup lang="ts">
import TopNav from "./TopNav.vue"
import NavSide from "./NavSide.vue"
</script>
