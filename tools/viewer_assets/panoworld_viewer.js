import * as THREE from "./vendor/three.module.min.js";

const manifest = window.PANOWORLD_MANIFEST;
const VIEWS = Array.isArray(manifest?.views) ? manifest.views : [];
const VIEWPOINT_MAP = new Map(VIEWS.map((node, index) => [String(node.id), { node, index }]));
const START_VIEWPOINT_ID = getInitialViewpointId();
const START_VIEWPOINT_TARGET_ID = VIEWPOINT_MAP.has("0016") ? "0016" : null;
const PANORAMA_CROSSFADE_DURATION_MS = 520;
const MAX_TEXTURE_CACHE_SIZE = 18;

function getInitialViewpointId() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("view") || params.get("v");
  if (requested && VIEWS.some((view) => String(view.id) === requested)) {
    return requested;
  }
  return VIEWS.length ? String(VIEWS[0].id) : "";
}

function clamp(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value));
}

function vecLength(vector) {
  return Math.hypot(vector[0], vector[1], vector[2]);
}

function normalizeVec(vector) {
  const length = vecLength(vector) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function subtractVec(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function addVec(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scaleVec(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function flattenRotation(rotation) {
  if (!Array.isArray(rotation)) {
    return null;
  }
  if (rotation.length === 9) {
    return rotation.map(Number);
  }
  if (rotation.length === 3 && rotation.every((row) => Array.isArray(row) && row.length >= 3)) {
    return [
      Number(rotation[0][0]), Number(rotation[0][1]), Number(rotation[0][2]),
      Number(rotation[1][0]), Number(rotation[1][1]), Number(rotation[1][2]),
      Number(rotation[2][0]), Number(rotation[2][1]), Number(rotation[2][2])
    ];
  }
  return null;
}

function getPosition(node) {
  const position = node?.pose?.position || node?.position;
  return Array.isArray(position) && position.length >= 3 ? position.map(Number).slice(0, 3) : null;
}

function getRotation(node) {
  return flattenRotation(node?.pose?.rotation || node?.rotation);
}

function multiplyMat3Vec3(m, v) {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
  ];
}

function multiplyMat3TransposeVec3(m, v) {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2]
  ];
}

function cameraToViewerVector(cameraVector) {
  return [cameraVector[2], -cameraVector[1], cameraVector[0]];
}

function viewerToCameraVector(viewerVector) {
  return [viewerVector[2], -viewerVector[1], viewerVector[0]];
}

function worldDirectionToViewerVector(node, worldDirection) {
  const rotation = getRotation(node);
  if (!rotation) {
    return normalizeVec(worldDirection);
  }
  return normalizeVec(cameraToViewerVector(multiplyMat3TransposeVec3(rotation, worldDirection)));
}

function viewerDirectionToWorldVector(node, viewerDirection) {
  const rotation = getRotation(node);
  if (!rotation) {
    return normalizeVec(viewerDirection);
  }
  return normalizeVec(multiplyMat3Vec3(rotation, viewerToCameraVector(viewerDirection)));
}

function viewerDirectionToHotspotVector(viewerDirection) {
  return [-viewerDirection[0], viewerDirection[1], -viewerDirection[2]];
}

function viewerDirectionToAngles(viewerDirection) {
  return {
    yaw: Math.atan2(viewerDirection[2], viewerDirection[0]),
    pitch: Math.asin(clamp(viewerDirection[1], -1, 1))
  };
}

function anglesToViewerDirection(yaw, pitch) {
  const cp = Math.cos(pitch);
  return [cp * Math.cos(yaw), Math.sin(pitch), cp * Math.sin(yaw)];
}

function wrapAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function computeDefaultAngles(node) {
  const currentPosition = getPosition(node);
  if (!currentPosition) {
    return { yaw: 0, pitch: 0 };
  }

  if (String(node.id) === START_VIEWPOINT_ID && START_VIEWPOINT_TARGET_ID) {
    const targetNode = VIEWPOINT_MAP.get(START_VIEWPOINT_TARGET_ID)?.node;
    const targetPosition = getPosition(targetNode);
    if (targetPosition) {
      const startDelta = subtractVec(targetPosition, currentPosition);
      const startAngles = viewerDirectionToAngles(worldDirectionToViewerVector(node, startDelta));
      return {
        yaw: wrapAngle(startAngles.yaw + Math.PI),
        pitch: startAngles.pitch
      };
    }
  }

  const others = VIEWS.filter((item) => String(item.id) !== String(node.id) && getPosition(item));
  if (!others.length) {
    return { yaw: 0, pitch: 0 };
  }

  const centroid = others.reduce((acc, item) => addVec(acc, getPosition(item)), [0, 0, 0]);
  const worldTarget = scaleVec(centroid, 1 / others.length);
  const delta = subtractVec(worldTarget, currentPosition);
  return viewerDirectionToAngles(worldDirectionToViewerVector(node, delta));
}

function formatShortViewpointId(id) {
  return String(Number.parseInt(id, 10)).padStart(2, "0");
}

function createHotspotTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;

  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);

  context.beginPath();
  context.arc(64, 64, 34, 0, Math.PI * 2);
  context.fillStyle = "rgba(255, 255, 255, 0.16)";
  context.fill();

  context.beginPath();
  context.arc(64, 64, 28, 0, Math.PI * 2);
  context.strokeStyle = "rgba(255, 255, 255, 0.95)";
  context.lineWidth = 6;
  context.stroke();

  context.beginPath();
  context.arc(64, 64, 10, 0, Math.PI * 2);
  context.fillStyle = "#ffffff";
  context.fill();

  return new THREE.CanvasTexture(canvas);
}

function initPanoramaTour() {
  const stage = document.getElementById("pano-tour-stage");
  if (!stage || !VIEWS.length) {
    return;
  }

  const loading = document.getElementById("pano-tour-loading");
  const currentLabel = document.getElementById("pano-tour-current");
  const hotspotOverlay = document.getElementById("pano-tour-hotspots");
  const tooltip = document.getElementById("pano-tour-tooltip");
  const strip = document.getElementById("pano-tour-strip");
  const fovLabel = document.getElementById("pano-tour-fov-label");

  let renderer;
  let scene;
  let camera;
  let sphereGeometry;
  let sphereA;
  let sphereB;
  let activeSphere;
  let standbySphere;
  let hotspotGroup;
  let hotspotTexture;
  let raycaster;
  let hoverHotspot = null;
  let hoverHotspotId = null;
  let currentViewpointId = START_VIEWPOINT_ID;
  let yaw = 0;
  let pitch = 0;
  let defaultAngles = { yaw: 0, pitch: 0 };
  let isPointerDown = false;
  let hasDragged = false;
  let pointerStartX = 0;
  let pointerStartY = 0;
  let pointerYaw = 0;
  let pointerPitch = 0;
  let transition = null;
  let isSwitching = false;
  let animationFrameId = null;
  const textureLoader = new THREE.TextureLoader();
  const textureEntries = new Map();
  const hotspotButtons = new Map();
  const stripButtons = new Map();
  const interactiveHotspotIds = new Set();
  const visibleHotspotIds = new Set();
  const pointer = new THREE.Vector2();
  const cameraDirection = new THREE.Vector3();
  const projectedPoint = new THREE.Vector3();

  function showLoading(message) {
    if (message) {
      loading.lastElementChild.textContent = message;
    }
    loading.classList.remove("is-hidden");
  }

  function hideLoading() {
    loading.classList.add("is-hidden");
  }

  function updateStripHighlightState() {
    stripButtons.forEach((button, buttonId) => {
      button.classList.toggle("is-active", buttonId === currentViewpointId);
    });
  }

  function setCurrentLabel(id) {
    currentLabel.textContent = "Viewpoint " + id;
    updateStripHighlightState();
  }

  function setHoveredHotspot(sprite, hotspotId) {
    hoverHotspot = sprite || null;
    hoverHotspotId = hotspotId || (sprite ? sprite.userData.id : null);
    updateHotspotHoverState();
  }

  function setTooltip(sprite) {
    setHoveredHotspot(sprite, sprite ? sprite.userData.id : null);
    tooltip.hidden = !sprite;
    stage.style.cursor = sprite ? "pointer" : (isPointerDown ? "grabbing" : "grab");
  }

  function updateTooltipPosition() {
    if (!hoverHotspot || tooltip.hidden) {
      return;
    }

    const projected = hoverHotspot.position.clone().project(camera);
    const rect = stage.getBoundingClientRect();
    const x = ((projected.x + 1) / 2) * rect.width;
    const y = ((-projected.y + 1) / 2) * rect.height;

    tooltip.textContent = "Go to viewpoint " + hoverHotspot.userData.id;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }

  function setPointerFromEvent(event) {
    const rect = stage.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function ensureHotspotButton(id) {
    if (hotspotButtons.has(id)) {
      return hotspotButtons.get(id);
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "pano-tour-hotspot-button";
    button.hidden = true;
    button.title = "Go to viewpoint " + id;
    button.setAttribute("aria-label", "Go to viewpoint " + id);
    button.innerHTML = [
      '<span class="pano-tour-hotspot-ring pano-tour-hotspot-ring--outer" aria-hidden="true"></span>',
      '<span class="pano-tour-hotspot-ring pano-tour-hotspot-ring--inner" aria-hidden="true"></span>',
      '<span class="pano-tour-hotspot-core" aria-hidden="true"></span>'
    ].join("");

    ["pointerdown", "pointermove", "pointerup"].forEach((eventName) => {
      button.addEventListener(eventName, (event) => {
        event.stopPropagation();
      });
    });

    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      switchViewpoint(id, true);
    });

    button.addEventListener("mouseenter", () => {
      setHoveredHotspot(null, id);
      stage.style.cursor = "pointer";
      tooltip.hidden = false;
      tooltip.textContent = "Go to viewpoint " + id;
      tooltip.style.left = button.style.left;
      tooltip.style.top = button.style.top;
    });

    button.addEventListener("mouseleave", () => {
      setHoveredHotspot(null, null);
      stage.style.cursor = isPointerDown ? "grabbing" : "grab";
      tooltip.hidden = true;
    });

    hotspotOverlay.appendChild(button);
    hotspotButtons.set(id, button);
    return button;
  }

  function applyCameraOrientation() {
    const direction = anglesToViewerDirection(yaw, pitch);
    camera.lookAt(direction[0], direction[1], direction[2]);
  }

  function updateFovLabel() {
    fovLabel.textContent = "FOV " + Math.round(camera.fov) + "°";
  }

  function setFov(nextFov) {
    camera.fov = clamp(nextFov, 42, 88);
    camera.updateProjectionMatrix();
    updateFovLabel();
  }

  function createScene() {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(stage.clientWidth, stage.clientHeight, false);
    renderer.setClearColor(0x0b1220, 1);
    stage.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(72, stage.clientWidth / stage.clientHeight, 0.1, 120);
    camera.position.set(0, 0, 0);
    camera.up.set(0, 1, 0);
    updateFovLabel();

    sphereGeometry = new THREE.SphereGeometry(50, 72, 48);
    sphereGeometry.scale(-1, 1, 1);

    const materialA = new THREE.MeshBasicMaterial({ transparent: true, opacity: 1, depthWrite: false });
    const materialB = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });

    sphereA = new THREE.Mesh(sphereGeometry, materialA);
    sphereB = new THREE.Mesh(sphereGeometry, materialB);
    sphereA.renderOrder = 0;
    sphereB.renderOrder = 1;
    sphereB.visible = false;

    scene.add(sphereA);
    scene.add(sphereB);

    activeSphere = sphereA;
    standbySphere = sphereB;

    hotspotGroup = new THREE.Group();
    hotspotGroup.renderOrder = 20;
    scene.add(hotspotGroup);

    hotspotTexture = createHotspotTexture();
    hotspotTexture.colorSpace = THREE.SRGBColorSpace;

    raycaster = new THREE.Raycaster();
  }

  function createHotspotSprite(targetNode, direction, distance) {
    const material = new THREE.SpriteMaterial({
      map: hotspotTexture,
      transparent: true,
      color: new THREE.Color("#7ad3ff"),
      depthTest: false,
      depthWrite: false
    });

    const sprite = new THREE.Sprite(material);
    const hotspotDirection = viewerDirectionToHotspotVector(direction);
    sprite.position.set(hotspotDirection[0], hotspotDirection[1], hotspotDirection[2]).multiplyScalar(9.5);
    sprite.renderOrder = 20;
    sprite.frustumCulled = false;
    const scale = clamp(1.64 - distance * 0.14, 0.8, 1.26);
    sprite.scale.set(scale, scale, scale);
    const buttonScale = clamp(1.58 - distance * 0.12, 0.62, 1.4);
    sprite.userData = {
      id: String(targetNode.id),
      baseScale: scale,
      buttonScale,
      distance
    };
    return sprite;
  }

  function rebuildHotspots() {
    while (hotspotGroup.children.length) {
      const child = hotspotGroup.children[0];
      hotspotGroup.remove(child);
      child.material.dispose();
    }

    const currentNode = VIEWPOINT_MAP.get(currentViewpointId)?.node;
    const currentPosition = getPosition(currentNode);
    if (!currentNode || !currentPosition) {
      return;
    }

    VIEWS.forEach((targetNode) => {
      if (String(targetNode.id) === currentViewpointId) {
        return;
      }
      const targetPosition = getPosition(targetNode);
      if (!targetPosition) {
        return;
      }

      const delta = subtractVec(targetPosition, currentPosition);
      const direction = worldDirectionToViewerVector(currentNode, delta);
      const distance = vecLength(delta);
      hotspotGroup.add(createHotspotSprite(targetNode, direction, distance));
    });
  }

  function updateHotspotHoverState() {
    if (!hotspotGroup) {
      return;
    }

    hotspotGroup.children.forEach((sprite) => {
      const isHovered = hoverHotspotId === sprite.userData.id;
      const targetScale = sprite.userData.baseScale * (isHovered ? 1.18 : 1);
      sprite.scale.set(targetScale, targetScale, targetScale);
      sprite.material.color.set(isHovered ? "#ffcf70" : "#7ad3ff");
    });
  }

  function updateHotspotButtons() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    const activeIds = new Set();
    const visibleEntries = [];
    const rootFontSize = parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
    const overlapBaseRadius = rootFontSize * 1.43;

    interactiveHotspotIds.clear();
    visibleHotspotIds.clear();

    camera.getWorldDirection(cameraDirection);

    hotspotGroup.children.forEach((sprite) => {
      const id = sprite.userData.id;
      const button = ensureHotspotButton(id);
      const facingScore = sprite.position.clone().normalize().dot(cameraDirection);
      projectedPoint.copy(sprite.position).project(camera);

      activeIds.add(id);

      const isVisible = (
        hotspotGroup.visible &&
        !transition &&
        facingScore > 0.04 &&
        projectedPoint.z > -1 &&
        projectedPoint.z < 1 &&
        projectedPoint.x > -1.12 &&
        projectedPoint.x < 1.12 &&
        projectedPoint.y > -1.12 &&
        projectedPoint.y < 1.12
      );

      if (!isVisible) {
        if (tooltip.textContent === "Go to viewpoint " + id) {
          tooltip.hidden = true;
        }
        button.hidden = true;
        button.style.pointerEvents = "none";
        button.tabIndex = -1;
        button.setAttribute("aria-disabled", "true");
        return;
      }

      button.hidden = false;
      const screenX = ((projectedPoint.x + 1) / 2) * width;
      const screenY = ((-projectedPoint.y + 1) / 2) * height;
      button.style.left = screenX + "px";
      button.style.top = screenY + "px";
      button.style.zIndex = String(1000 - Math.round(sprite.userData.distance * 100));
      button.style.setProperty("--pano-hotspot-scale", String(sprite.userData.buttonScale));

      visibleHotspotIds.add(id);
      visibleEntries.push({
        id,
        button,
        sprite,
        distance: sprite.userData.distance,
        x: screenX,
        y: screenY,
        radius: overlapBaseRadius * sprite.userData.buttonScale
      });
    });

    visibleEntries.sort((a, b) => a.distance - b.distance);

    const clickableEntries = [];
    visibleEntries.forEach((entry) => {
      const blockedByNearer = clickableEntries.some((otherEntry) => {
        const dx = entry.x - otherEntry.x;
        const dy = entry.y - otherEntry.y;
        return Math.hypot(dx, dy) < entry.radius + otherEntry.radius;
      });

      entry.button.style.pointerEvents = blockedByNearer ? "none" : "auto";
      entry.button.tabIndex = blockedByNearer ? -1 : 0;
      entry.button.setAttribute("aria-disabled", blockedByNearer ? "true" : "false");

      if (blockedByNearer) {
        if (hoverHotspotId === entry.id) {
          setTooltip(null);
        }
        return;
      }

      interactiveHotspotIds.add(entry.id);
      clickableEntries.push(entry);
    });

    hotspotButtons.forEach((button, id) => {
      if (!activeIds.has(id)) {
        button.hidden = true;
        button.style.pointerEvents = "none";
        button.tabIndex = -1;
        button.setAttribute("aria-disabled", "true");
      }
    });
  }

  function updateDefaultView() {
    const node = VIEWPOINT_MAP.get(currentViewpointId)?.node;
    defaultAngles = node ? computeDefaultAngles(node) : { yaw: 0, pitch: 0 };
  }

  function textureKey(nodeId) {
    return String(nodeId);
  }

  function loadTextureForNode(nodeId) {
    const entry = VIEWPOINT_MAP.get(String(nodeId));
    if (!entry) {
      return Promise.reject(new Error("Unknown viewpoint: " + nodeId));
    }
    const key = textureKey(nodeId);
    const existing = textureEntries.get(key);
    if (existing && existing.texture) {
      existing.lastUsed = performance.now();
      return Promise.resolve(existing.texture);
    }
    if (existing && existing.promise) {
      return existing.promise;
    }

    const cacheEntry = existing || { texture: null, promise: null, lastUsed: performance.now() };
    cacheEntry.promise = textureLoader.loadAsync(entry.node.image).then((texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      cacheEntry.texture = texture;
      cacheEntry.promise = null;
      cacheEntry.lastUsed = performance.now();
      return texture;
    });
    textureEntries.set(key, cacheEntry);
    return cacheEntry.promise;
  }

  function pruneTextureCache() {
    if (textureEntries.size <= MAX_TEXTURE_CACHE_SIZE) {
      return;
    }

    const entries = Array.from(textureEntries.entries()).sort((a, b) => b[1].lastUsed - a[1].lastUsed);
    entries.forEach(([key, entry], index) => {
      if (key === currentViewpointId || index < MAX_TEXTURE_CACHE_SIZE || !entry.texture) {
        return;
      }
      entry.texture.dispose();
      textureEntries.delete(key);
    });
  }

  function prefetchViewpoints() {
    if (!("requestIdleCallback" in window)) {
      window.setTimeout(prefetchViewpoints, 1200);
      return;
    }
    window.requestIdleCallback(() => {
      VIEWS.forEach((node, index) => {
        if (String(node.id) === currentViewpointId) {
          return;
        }
        window.setTimeout(() => {
          loadTextureForNode(node.id).catch(() => {});
        }, index * 170);
      });
    }, { timeout: 1600 });
  }

  function computePreservedAngles(nextNodeId) {
    const currentNode = VIEWPOINT_MAP.get(currentViewpointId)?.node;
    const nextNode = VIEWPOINT_MAP.get(String(nextNodeId))?.node;
    if (!currentNode || !nextNode) {
      return { yaw, pitch };
    }
    const worldDirection = viewerDirectionToWorldVector(currentNode, anglesToViewerDirection(yaw, pitch));
    return viewerDirectionToAngles(worldDirectionToViewerVector(nextNode, worldDirection));
  }

  function startCrossFade(texture, nextNodeId, nextAngles) {
    standbySphere.material.map = texture;
    standbySphere.material.needsUpdate = true;
    standbySphere.material.opacity = 0;
    standbySphere.visible = true;

    currentViewpointId = String(nextNodeId);
    yaw = nextAngles.yaw;
    pitch = nextAngles.pitch;
    applyCameraOrientation();
    updateDefaultView();
    setCurrentLabel(currentViewpointId);
    updateBrowserUrl(currentViewpointId);

    setTooltip(null);
    hotspotGroup.visible = false;

    transition = {
      startedAt: performance.now(),
      duration: PANORAMA_CROSSFADE_DURATION_MS,
      from: activeSphere,
      to: standbySphere
    };
  }

  function updateBrowserUrl(nodeId) {
    const url = new URL(window.location.href);
    url.searchParams.set("v", nodeId);
    window.history.replaceState(null, "", url);
  }

  async function switchViewpoint(nextNodeId, preserveOrientation) {
    const nextId = String(nextNodeId);
    if (isSwitching || nextId === currentViewpointId || !VIEWPOINT_MAP.has(nextId)) {
      return;
    }

    isSwitching = true;
    showLoading("Loading viewpoint " + nextId + "...");

    const nextAngles = preserveOrientation ? computePreservedAngles(nextId) : computeDefaultAngles(VIEWPOINT_MAP.get(nextId).node);
    try {
      const texture = await loadTextureForNode(nextId);
      hideLoading();
      startCrossFade(texture, nextId, nextAngles);
    } catch (error) {
      console.error(error);
      showLoading("Failed to load viewpoint " + nextId);
      window.setTimeout(hideLoading, 1400);
      isSwitching = false;
    }
  }

  function handleClick(event) {
    setPointerFromEvent(event);
    raycaster.setFromCamera(pointer, camera);
    const intersections = raycaster.intersectObjects(hotspotGroup.children, false);
    const interactiveIntersection = intersections.find((intersection) => (
      visibleHotspotIds.has(intersection.object.userData.id) &&
      interactiveHotspotIds.has(intersection.object.userData.id)
    ));
    if (interactiveIntersection) {
      switchViewpoint(interactiveIntersection.object.userData.id, true);
    }
  }

  function updateHoverFromEvent(event) {
    if (isPointerDown || transition) {
      setTooltip(null);
      return;
    }

    setPointerFromEvent(event);
    raycaster.setFromCamera(pointer, camera);
    const intersections = raycaster.intersectObjects(hotspotGroup.children, false);
    const interactiveIntersection = intersections.find((intersection) => (
      visibleHotspotIds.has(intersection.object.userData.id) &&
      interactiveHotspotIds.has(intersection.object.userData.id)
    ));
    setTooltip(interactiveIntersection ? interactiveIntersection.object : null);
  }

  function resize() {
    if (!renderer || !camera) {
      return;
    }
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height, false);
  }

  function animate(now) {
    animationFrameId = window.requestAnimationFrame(animate);

    if (transition) {
      const progress = clamp((now - transition.startedAt) / transition.duration, 0, 1);
      const eased = 0.5 - 0.5 * Math.cos(progress * Math.PI);
      transition.from.material.opacity = 1 - eased;
      transition.to.material.opacity = eased;

      if (progress >= 1) {
        transition.from.material.opacity = 0;
        transition.from.visible = false;
        transition.to.material.opacity = 1;
        transition.to.visible = true;
        activeSphere = transition.to;
        standbySphere = transition.from;
        transition = null;
        rebuildHotspots();
        hotspotGroup.visible = true;
        isSwitching = false;
        pruneTextureCache();
      }
    }

    updateHotspotHoverState();
    updateHotspotButtons();
    updateTooltipPosition();
    renderer.render(scene, camera);
  }

  function bindEvents() {
    stage.addEventListener("pointerdown", (event) => {
      if (isSwitching) {
        return;
      }
      isPointerDown = true;
      hasDragged = false;
      pointerStartX = event.clientX;
      pointerStartY = event.clientY;
      pointerYaw = yaw;
      pointerPitch = pitch;
      stage.classList.add("is-dragging");
      stage.setPointerCapture(event.pointerId);
      setTooltip(null);
    });

    stage.addEventListener("pointermove", (event) => {
      if (isPointerDown) {
        const dx = event.clientX - pointerStartX;
        const dy = event.clientY - pointerStartY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
          hasDragged = true;
        }
        yaw = pointerYaw - dx * 0.0055;
        pitch = clamp(pointerPitch + dy * 0.0036, -1.2, 1.2);
        applyCameraOrientation();
      } else {
        updateHoverFromEvent(event);
      }
    });

    stage.addEventListener("pointerup", (event) => {
      if (!isPointerDown) {
        return;
      }
      stage.releasePointerCapture(event.pointerId);
      stage.classList.remove("is-dragging");
      isPointerDown = false;

      if (!hasDragged && !isSwitching) {
        handleClick(event);
      } else {
        updateHoverFromEvent(event);
      }
    });

    stage.addEventListener("pointerleave", () => {
      if (!isPointerDown) {
        setTooltip(null);
      }
    });

    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      setFov(camera.fov + event.deltaY * 0.018);
    }, { passive: false });

    window.addEventListener("resize", resize);
  }

  function bindControls() {
    const prevButton = document.getElementById("pano-tour-prev");
    const nextButton = document.getElementById("pano-tour-next");
    const resetButton = document.getElementById("pano-tour-reset");
    const zoomOutButton = document.getElementById("pano-tour-zoom-out");
    const zoomInButton = document.getElementById("pano-tour-zoom-in");
    const fullButton = document.getElementById("pano-tour-fullscreen");

    [prevButton, nextButton, resetButton, zoomOutButton, zoomInButton, fullButton].forEach((button) => {
      ["pointerdown", "pointermove", "pointerup", "click", "wheel"].forEach((eventName) => {
        button.addEventListener(eventName, (event) => {
          event.stopPropagation();
        });
      });
    });

    prevButton.addEventListener("click", () => {
      const currentIndex = VIEWPOINT_MAP.get(currentViewpointId)?.index || 0;
      switchViewpoint(VIEWS[(currentIndex - 1 + VIEWS.length) % VIEWS.length].id, true);
    });
    nextButton.addEventListener("click", () => {
      const currentIndex = VIEWPOINT_MAP.get(currentViewpointId)?.index || 0;
      switchViewpoint(VIEWS[(currentIndex + 1) % VIEWS.length].id, true);
    });
    resetButton.addEventListener("click", () => {
      yaw = defaultAngles.yaw;
      pitch = defaultAngles.pitch;
      setFov(72);
      applyCameraOrientation();
    });
    zoomOutButton.addEventListener("click", () => setFov(camera.fov + 6));
    zoomInButton.addEventListener("click", () => setFov(camera.fov - 6));
    fullButton.addEventListener("click", () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") yaw += 0.08;
      if (event.key === "ArrowRight") yaw -= 0.08;
      if (event.key === "ArrowUp") pitch = Math.min(1.2, pitch + 0.08);
      if (event.key === "ArrowDown") pitch = Math.max(-1.2, pitch - 0.08);
      if (event.key === "+" || event.key === "=") setFov(camera.fov - 6);
      if (event.key === "-" || event.key === "_") setFov(camera.fov + 6);
      if (event.key === "r" || event.key === "R") {
        yaw = defaultAngles.yaw;
        pitch = defaultAngles.pitch;
        setFov(72);
      }
      applyCameraOrientation();
    });
  }

  function buildStrip() {
    const fragment = document.createDocumentFragment();
    VIEWS.forEach((node) => {
      const id = String(node.id);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pano-tour-chip";
      button.textContent = formatShortViewpointId(id);
      button.title = "Go to viewpoint " + id;
      button.setAttribute("aria-label", "Go to viewpoint " + id);
      ["pointerdown", "pointermove", "pointerup", "click", "wheel"].forEach((eventName) => {
        button.addEventListener(eventName, (event) => {
          event.stopPropagation();
        });
      });
      button.addEventListener("click", () => switchViewpoint(id, true));
      stripButtons.set(id, button);
      fragment.appendChild(button);
    });
    strip.innerHTML = "";
    strip.appendChild(fragment);
  }

  async function boot() {
    createScene();
    bindEvents();
    bindControls();
    buildStrip();
    resize();

    const startNode = VIEWPOINT_MAP.get(START_VIEWPOINT_ID)?.node || VIEWS[0];
    currentViewpointId = String(startNode.id);
    defaultAngles = computeDefaultAngles(startNode);
    yaw = defaultAngles.yaw;
    pitch = defaultAngles.pitch;
    applyCameraOrientation();

    showLoading("Loading panorama tour...");
    try {
      const texture = await loadTextureForNode(currentViewpointId);
      activeSphere.material.map = texture;
      activeSphere.material.needsUpdate = true;
      activeSphere.visible = true;
      rebuildHotspots();
      setCurrentLabel(currentViewpointId);
      hideLoading();
      prefetchViewpoints();
      animate(performance.now());
    } catch (error) {
      console.error(error);
      showLoading("Failed to initialize the panorama tour.");
    }
  }

  boot();

  window.addEventListener("beforeunload", () => {
    if (animationFrameId !== null) {
      window.cancelAnimationFrame(animationFrameId);
    }
  });
}

document.addEventListener("DOMContentLoaded", initPanoramaTour);
