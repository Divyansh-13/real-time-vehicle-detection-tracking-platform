# ============================================================================
# Frontend Dockerfile — Multi-stage build
# ============================================================================
# Stage 1: Build React app with Node.js
# Stage 2: Serve with Nginx (tiny image)
#
# Build: docker build -t vehicle-frontend -f frontend/Dockerfile .
# Run:   docker run -p 3000:80 vehicle-frontend
# ============================================================================

# ── Build stage ──
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies (cached layer)
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent

# Copy source and build
COPY frontend/ .
RUN npm run build

# ── Serve stage ──
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config for SPA routing
COPY deployment/nginx/frontend.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q --spider http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]
