// panels.jsx — Camera feed, audio beam viz, haptic log, status bar, info row, modal

const { useEffect, useRef, useState, useMemo } = React;

// ─────────────────────────────────────────────
// Camera feed (simulated tactical view + YOLO bboxes from threats in FOV)
// ─────────────────────────────────────────────
function CameraFeed({ threats, yoloDets, fps, yoloMs, temp, heading, liveCam, onLiveCamToggle, liveConnected }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(0);
  const detsRef   = useRef(yoloDets || []);
  const liveRef   = useRef(liveConnected);
  const headingRef = useRef(heading);
  useEffect(()=>{ detsRef.current = yoloDets || []; }, [yoloDets]);
  useEffect(()=>{ liveRef.current = liveConnected; }, [liveConnected]);
  useEffect(()=>{ headingRef.current = heading; }, [heading]);
  const _feedHost = window.location.hostname || "raspberrypi.local";
  const _mjpegUrl = `http://${_feedHost}:8090/video_feed`;

  useEffect(()=>{
    const c = canvasRef.current; if(!c) return;
    function tick(){
      const dpr = window.devicePixelRatio||1;
      const rect = c.getBoundingClientRect();
      const W = Math.max(10, rect.width), H = Math.max(10, rect.height);
      if(c.width !== Math.round(W*dpr)){ c.width = Math.round(W*dpr); c.height = Math.round(H*dpr); }
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,W,H);

      const dets = detsRef.current;
      const isLive = liveRef.current;
      const hdg = headingRef.current;

      if(!isLive){
        ctx.fillStyle="#f7fafc"; ctx.fillRect(0,0,W,H);
        ctx.fillStyle="rgba(15,157,138,0.42)";
        ctx.font="11px Inter, system-ui, sans-serif";
        ctx.textAlign="center"; ctx.textBaseline="middle";
        ctx.fillText("NO FEED — PIPELINE OFFLINE",W/2,H/2);
        ctx.textAlign="left"; ctx.textBaseline="alphabetic";
        animRef.current = requestAnimationFrame(tick);
        return;
      }

      // ── YOLO bboxes ──
      const URGENCY_COL = {critical:"#E24B4A", high:"#F59E0B", medium:"#20C997", low:"rgba(32,201,151,0.5)"};
      ctx.font = "bold 10px JetBrains Mono, monospace";

      dets.forEach(d=>{
        const [x1n,y1n,x2n,y2n] = d.bbox_n;
        const bx=x1n*W, by=y1n*H, bw=(x2n-x1n)*W, bh=(y2n-y1n)*H;
        const col = URGENCY_COL[d.urgency] || "#20C997";
        const brk = Math.min(16, Math.max(6, bw*0.22));
        ctx.strokeStyle = col; ctx.lineWidth = 1.8;
        [[bx,by,1,1],[bx+bw,by,-1,1],[bx,by+bh,1,-1],[bx+bw,by+bh,-1,-1]].forEach(([x,y,sx,sy])=>{
          ctx.beginPath();
          ctx.moveTo(x+brk*sx,y); ctx.lineTo(x,y); ctx.lineTo(x,y+brk*sy);
          ctx.stroke();
        });
        const lbl = `${d.label}  ${Math.round(d.conf*100)}%`;
        const tw = ctx.measureText(lbl).width;
        ctx.fillStyle="rgba(255,255,255,0.84)"; ctx.fillRect(bx, by-15, tw+10, 14);
        ctx.fillStyle=col; ctx.fillText(lbl, bx+5, by-4);
        if(d.dist > 0){
          const dt = `${d.dist}m`;
          const dw = ctx.measureText(dt).width;
          ctx.fillStyle="rgba(255,255,255,0.84)"; ctx.fillRect(bx+bw-dw-8, by+bh+1, dw+8, 13);
          ctx.fillStyle=col; ctx.fillText(dt, bx+bw-dw-3, by+bh+11);
        }
      });

      // ── Navigation overlay (heading from Mac planner) ──
      const hasNav = Math.abs(hdg) > 1;
      if(hasNav){
        const now = performance.now();
        // heading: 0=forward(up), 90=right, -90 or 270=left, 180=back
        // Normalize to -180..180 relative to forward
        const relHdg = ((hdg + 180) % 360) - 180;

        // ── Top compass strip ──
        const stripH = 28;
        const stripGrad = ctx.createLinearGradient(0, 0, 0, stripH);
        stripGrad.addColorStop(0,   "rgba(7,16,22,0.42)");
        stripGrad.addColorStop(1,   "rgba(7,16,22,0)");
        ctx.fillStyle = stripGrad; ctx.fillRect(0, 0, W, stripH);

        // Cardinal marks on strip
        const cardinals = [
          ["N", 0], ["NE", 45], ["E", 90], ["SE", 135],
          ["S", 180], ["SW", 225], ["W", 270], ["NW", 315],
        ];
        const stripScale = W / 180; // how many px per degree in view
        const cx2 = W / 2;
        ctx.font = "600 9px Inter, system-ui, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        cardinals.forEach(([label, deg])=>{
          // position relative to heading (forward at center); hdg already 0-360 from server
          let offset = deg - hdg;
          if(offset > 180)  offset -= 360;
          if(offset < -180) offset += 360;
          const x = cx2 + offset * stripScale;
          if(x < -10 || x > W+10) return;
          const isMajor = label.length === 1;
          ctx.fillStyle = isMajor ? "rgba(15,157,138,0.92)" : "rgba(15,157,138,0.42)";
          ctx.fillText(label, x, 5);
          // tick mark
          ctx.strokeStyle = isMajor ? "rgba(15,157,138,0.55)" : "rgba(15,157,138,0.22)";
          ctx.lineWidth = isMajor ? 1.5 : 0.8;
          ctx.beginPath(); ctx.moveTo(x, 16); ctx.lineTo(x, 22); ctx.stroke();
        });

        // Center heading marker (inverted triangle)
        ctx.fillStyle = "#0f9d8a";
        ctx.beginPath();
        ctx.moveTo(cx2, 16); ctx.lineTo(cx2-5, 6); ctx.lineTo(cx2+5, 6); ctx.closePath(); ctx.fill();

        // Heading label
        const hdgLabel = Math.round(((hdg%360)+360)%360).toString().padStart(3,"0")+"°";
        ctx.font = "700 10px JetBrains Mono, monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const tw2 = ctx.measureText(hdgLabel).width;
        ctx.fillStyle = "rgba(255,255,255,0.72)";
        ctx.beginPath();
        if(ctx.roundRect) ctx.roundRect(cx2-tw2/2-6, 18, tw2+12, 16, 3);
        else ctx.rect(cx2-tw2/2-6, 18, tw2+12, 16);
        ctx.fill();
        ctx.fillStyle = "#1f2937";
        ctx.fillText(hdgLabel, cx2, 26);

        // ── AR ground path — drawn as filled trapezoid with left+right edges ──
        // Path starts BELOW screen so it enters the frame already wide.
        // Edges fan out toward camera, converge to vanishing point at horizon.
        const botY   = H * 0.92;
        const sinR   = Math.sin(relHdg * Math.PI / 180);
        const horizY = H * 0.55;  // vanishing point — path ends above midpoint
        const z0     = 0.3;       // near clip — below screen, enters frame wide

        // Gradient vignette
        const botGrad = ctx.createLinearGradient(0, H*0.55, 0, H);
        botGrad.addColorStop(0, "rgba(0,0,0,0)");
        botGrad.addColorStop(1, "rgba(31,41,55,0.60)");
        ctx.fillStyle = botGrad; ctx.fillRect(0, H*0.55, W, H*0.45);

        // Project world point {lat, dep} → screen {x, y, hw}
        // hw = screen half-width of path at this depth (0.4m world half-width)
        const HALF_W_M = 0.38;
        const LAT_SCALE = W * 0.55;
        function proj(pt) {
          const z  = Math.max(0.01, pt.dep);
          const sc = z0 / z;
          const y  = H - (H - horizY) * (1 - sc);
          return { x: W/2 + pt.lat * LAT_SCALE * sc, y, sc,
                   hw: HALF_W_M * LAT_SCALE * sc };
        }

        // Centerline waypoints — below-screen start so path is already wide at frame edge
        const worldPts = [
          { lat: 0,             dep: 0.3  },   // below screen
          { lat: -sinR * 0.40,  dep: 0.8  },   // near — counter-lean
          { lat:  sinR * 1.20,  dep: 2.5  },   // mid — main dodge
          { lat:  sinR * 2.60,  dep: 6.0  },   // far — locked heading
        ];
        const ns = worldPts.map(proj);

        // Deflect waypoints around YOLO bboxes — path must not pass through obstacles.
        // bbox_n coords are relative to the 224px YOLO frame; canvas may differ in size.
        // We treat the canvas W×H as the mapping target directly.
        {
          const dets = detsRef.current || [];
          const MARGIN = 1.4; // path half-width multiplier for clearance
          for (let i = 1; i < ns.length; i++) {
            const pt = ns[i];
            for (const det of dets) {
              const bb = det.bbox_n; if (!bb || bb.length < 4) continue;
              const bx1=bb[0]*W, by1=bb[1]*H, bx2=bb[2]*W, by2=bb[3]*H;
              const margin = (pt.hw || 10) * MARGIN;
              if (pt.y < by1 || pt.y > by2 + margin) continue; // wrong Y band
              if (pt.x + margin < bx1 || pt.x - margin > bx2) continue; // no X overlap
              // Blocked — push to whichever side has more room
              const goLeft  = bx1 - margin;
              const goRight = bx2 + margin;
              const spaceL  = goLeft;
              const spaceR  = W - goRight;
              pt.x = spaceL > spaceR ? goLeft : goRight;
            }
          }
        }

        // Build left + right edge points (perpendicular to centerline at each point)
        function perp(pts, i) {
          const a = pts[Math.max(0, i-1)], b = pts[Math.min(pts.length-1, i+1)];
          const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
          return { px: -dy/len, py: dx/len };
        }
        const left  = ns.map((p,i) => { const {px,py}=perp(ns,i); return {x:p.x+px*p.hw, y:p.y+py*p.hw}; });
        const right = ns.map((p,i) => { const {px,py}=perp(ns,i); return {x:p.x-px*p.hw, y:p.y-py*p.hw}; });

        // ① Filled lane — polygon of left edges then right edges reversed
        ctx.beginPath();
        ctx.moveTo(left[0].x, left[0].y);
        for(let i=1;i<left.length;i++) ctx.lineTo(left[i].x, left[i].y);
        for(let i=right.length-1;i>=0;i--) ctx.lineTo(right[i].x, right[i].y);
        ctx.closePath();
        const laneGrad = ctx.createLinearGradient(0, ns[ns.length-1].y, 0, H);
        laneGrad.addColorStop(0, "rgba(15,157,138,0.06)");
        laneGrad.addColorStop(1, "rgba(15,157,138,0.18)");
        ctx.fillStyle = laneGrad; ctx.fill();

        // ② Edge lines (left + right) — teal, fading toward horizon
        [left, right].forEach(edge => {
          ctx.beginPath(); ctx.moveTo(edge[0].x, edge[0].y);
          for(let i=1;i<edge.length;i++) ctx.lineTo(edge[i].x, edge[i].y);
          ctx.strokeStyle = "rgba(15,157,138,0.55)";
          ctx.lineWidth = 1.5; ctx.setLineDash([5,5]);
          ctx.lineDashOffset = -((now/60)%10);
          ctx.stroke();
        });
        ctx.setLineDash([]); ctx.lineDashOffset = 0;

        // ③ Dashed centerline — perspective-scaled per segment
        for(let i=0;i<ns.length-1;i++) {
          const a=ns[i], b=ns[i+1], avgSc=(a.sc+b.sc)/2;
          const lw=Math.min(5,Math.max(1.5,avgSc*6));
          const dash=Math.max(4,avgSc*20), gap=Math.max(3,avgSc*14);
          ctx.strokeStyle="rgba(15,157,138,0.95)"; ctx.lineWidth=lw; ctx.lineCap="round";
          ctx.setLineDash([dash,gap]);
          ctx.lineDashOffset=-((now/32)%(dash+gap));
          ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
        ctx.setLineDash([]); ctx.lineDashOffset=0;

        // ④ Arrowhead at far end
        const tip=ns[ns.length-1], tipPrv=ns[ns.length-2];
        const tipA=Math.atan2(tip.y-tipPrv.y, tip.x-tipPrv.x);
        const ah=Math.max(7,tip.sc*16), ahA=0.44;
        ctx.fillStyle="#0f9d8a"; ctx.strokeStyle="rgba(255,255,255,0.5)"; ctx.lineWidth=1.5;
        ctx.beginPath();
        ctx.moveTo(tip.x,tip.y);
        ctx.lineTo(tip.x-Math.cos(tipA-ahA)*ah, tip.y-Math.sin(tipA-ahA)*ah);
        ctx.lineTo(tip.x-Math.cos(tipA)*ah*0.38, tip.y-Math.sin(tipA)*ah*0.38);
        ctx.lineTo(tip.x-Math.cos(tipA+ahA)*ah, tip.y-Math.sin(tipA+ahA)*ah);
        ctx.closePath(); ctx.fill(); ctx.stroke();

        // Turn direction label at bottom
        const turnLabel = Math.abs(relHdg) < 15 ? "GO STRAIGHT"
          : relHdg > 0 ? `BEAR RIGHT ${Math.round(relHdg)}°`
          : `BEAR LEFT ${Math.round(-relHdg)}°`;
        ctx.font = "700 11px Inter, system-ui, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const tlw = ctx.measureText(turnLabel).width;
        ctx.fillStyle = "rgba(255,255,255,0.78)";
        ctx.beginPath();
        if(ctx.roundRect) ctx.roundRect(cx2-tlw/2-10, botY-11, tlw+20, 22, 5);
        else ctx.rect(cx2-tlw/2-10, botY-11, tlw+20, 22);
        ctx.fill();
        ctx.fillStyle = "#1f2937";
        ctx.fillText(turnLabel, cx2, botY);
      }

      animRef.current = requestAnimationFrame(tick);
    }
    animRef.current = requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(animRef.current);
  },[]);

  const latClass = yoloMs<100?"ok":yoloMs<200?"warn":"err";

  return (
    <div className="cell">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>Camera · YOLO v8-N</div>
        <div className="meta">
          <span>·</span>
          <button
            onClick={()=>onLiveCamToggle && onLiveCamToggle(!liveCam)}
            style={{
              background:"transparent",
              border:".5px solid "+(liveCam?"var(--teal)":"var(--div-hi)"),
              color: liveCam?"var(--teal)":"var(--tx-2)",
              fontFamily:"var(--mono)",fontSize:9,letterSpacing:".18em",
              padding:"2px 7px",cursor:"pointer",borderRadius:2
            }}>
            {liveCam ? "● OP-CAM" : "○ OP-CAM"}
          </button>
          <span>·</span>
          <span className={liveConnected?"live":""} style={{color:liveConnected?"var(--teal)":"var(--tx-3)"}}>
            {liveConnected?"● LIVE":"○ OFFLINE"}
          </span>
        </div>
      </div>
      {/* Video container: MJPEG behind exact-bbox canvas overlay */}
      <div className="cam" style={{margin:"10px", position:"relative"}}>
        <div style={{position:"relative",width:"100%",aspectRatio:"1/1",background:"#071016",overflow:"hidden",borderRadius:14}}>
          {liveConnected && (
            <img src={_mjpegUrl} style={{
              position:"absolute",inset:0,width:"100%",height:"100%",
              objectFit:"cover",
              zIndex:1,
              opacity:1,
              filter:"contrast(1.18) saturate(1.15) brightness(0.96)",
            }} alt="" />
          )}
          <canvas ref={canvasRef} style={{
            position:"absolute",inset:0,width:"100%",height:"100%",zIndex:2,
          }} />
          {/* HUD overlay */}
          <div style={{position:"absolute",top:6,left:8,display:"flex",gap:8,zIndex:3,fontFamily:"var(--mono)",fontSize:9,color:"var(--teal)",background:"rgba(7,16,22,0.46)",padding:"3px 6px",borderRadius:6}}>
            <span>{fps}fps</span>
            <span style={{color:"var(--div-hi)"}}>|</span>
            <span className={"lat "+latClass}>{yoloMs}ms</span>
            <span style={{color:"var(--div-hi)"}}>|</span>
            <span>{temp}°C</span>
          </div>
          <div style={{position:"absolute",top:6,right:8,zIndex:3,fontFamily:"var(--mono)",fontSize:9,color:"var(--red)",animation:"blink 1.2s infinite",background:"rgba(7,16,22,0.46)",padding:"3px 6px",borderRadius:6}}>● REC</div>
          <div style={{position:"absolute",bottom:5,left:0,right:0,display:"flex",justifyContent:"space-between",padding:"0 8px",zIndex:3,fontFamily:"var(--mono)",fontSize:8,color:"rgba(255,255,255,0.72)",textShadow:"0 1px 2px rgba(0,0,0,0.8)"}}>
            <span>CAM-0</span>
            <span>{liveConnected?`${(yoloDets||[]).length} DETS`:"OFFLINE"}</span>
            <span>YOLO·n</span>
          </div>
        </div>
        <LiveCamPiP
          enabled={!!liveCam}
          onClose={()=>onLiveCamToggle && onLiveCamToggle(false)}
          onRequestEnable={()=>onLiveCamToggle && onLiveCamToggle(true)}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Audio Beam Visualizer (bars + polar)
// ─────────────────────────────────────────────
function AudioBeams({ beams, mode }) {
  const polarRef = useRef(null);
  useEffect(()=>{
    if(mode!=="polar") return;
    const c = polarRef.current; if(!c) return;
    const dpr = window.devicePixelRatio||1;
    const W = 220, H = 140;
    if(c.width!==W*dpr){ c.width=W*dpr; c.height=H*dpr; c.style.width=W+"px"; c.style.height=H+"px"; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,H);
    const cx=W/2, cy=H*0.7, R=Math.min(W*0.4, H*0.6);
    // grid
    ctx.strokeStyle="rgba(200,155,40,0.10)"; ctx.lineWidth=0.5;
    [0.33,0.66,1].forEach(f=>{ ctx.beginPath(); ctx.arc(cx,cy,R*f,Math.PI,0); ctx.stroke(); });
    // axes
    ctx.strokeStyle="rgba(200,155,40,0.07)";
    [180,225,270,315,360].forEach(d=>{
      const a = d*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(cx,cy);
      ctx.lineTo(cx+Math.cos(a)*R, cy+Math.sin(a)*R); ctx.stroke();
    });
    // 4 beams: front=0(up), right=90, back=180(down), left=270
    const bs = [
      ["FRONT", beams.front, -90],
      ["RIGHT", beams.right, 0],
      ["LEFT",  beams.left, -180],
      ["BACK",  beams.back, 90],
    ];
    bs.forEach(([nm,e,deg])=>{
      const a = deg*Math.PI/180;
      const len = R*e;
      const hot = e>0.3;
      ctx.strokeStyle = hot ? "#0f9d8a" : "rgba(15,157,138,0.35)";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx+Math.cos(a)*len, cy+Math.sin(a)*len);
      ctx.stroke();
      // tip dot
      ctx.fillStyle = hot ? "#28b5d4" : "rgba(114,129,149,0.45)";
      ctx.beginPath();
      ctx.arc(cx+Math.cos(a)*len, cy+Math.sin(a)*len, 3,0,Math.PI*2);
      ctx.fill();
      // label
      ctx.fillStyle = "rgba(185,155,70,0.70)";
      ctx.font = "9px 'JetBrains Mono',monospace";
      ctx.textAlign="center";
      ctx.fillText(nm, cx+Math.cos(a)*(R+10), cy+Math.sin(a)*(R+10)+3);
    });
    // center
    ctx.fillStyle = "#0f9d8a";
    ctx.beginPath(); ctx.arc(cx,cy,3,0,Math.PI*2); ctx.fill();
  },[beams, mode]);

  return (
    <div className="cell">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>Audio Beamform</div>
        <div className="meta">
          <span>MAX4466 ×4</span>
          <span>·</span>
          <span className="live">20HZ</span>
        </div>
      </div>
      {mode==="polar" ? (
        <div className="beam-polar">
          <canvas ref={polarRef} />
        </div>
      ) : (
        <div className="beams">
          {[
            ["FRONT", beams.front],
            ["RIGHT", beams.right],
            ["BACK",  beams.back],
            ["LEFT",  beams.left],
          ].map(([nm,e])=>(
            <div key={nm} className="row">
              <div className="nm">{nm}</div>
              <div className="bar"><div className={"fill "+(e>0.4?"hot":"")} style={{width: (e*100).toFixed(0)+"%"}}/></div>
              <div className="pct">{Math.round(e*100)}%</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Haptic Event Log
// ─────────────────────────────────────────────
function HapticLog({ events, expanded, setExpanded, onAck }) {
  const listRef = useRef(null);
  const stickRef = useRef(true);
  useEffect(()=>{
    const el = listRef.current; if(!el) return;
    function onScroll(){
      const near = el.scrollTop < 8;
      stickRef.current = near;
    }
    el.addEventListener("scroll", onScroll);
    return ()=>el.removeEventListener("scroll", onScroll);
  },[]);
  useEffect(()=>{
    if(stickRef.current && listRef.current){
      listRef.current.scrollTop = 0;
    }
  },[events.length]);

  return (
    <div className="cell area-log">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>Event Log</div>
        <div className="meta">
          <span>{events.length} EVT</span>
          <span>·</span>
          <span className="live">RT</span>
        </div>
      </div>
      <div className="log-list" ref={listRef}>
        {events.map(ev=>(
          <div key={ev.id}
            className={"log-row "+(ev.urgency==="critical"?"crit ":"")+(expanded===ev.id?"expanded":"")}
            onClick={()=>setExpanded(expanded===ev.id?null:ev.id)}>
            <div className="ln1">
              <span className="time">{hhmmss(ev.bornAt)}</span>
              <span className="dir">{arrowFor(ev.angle)} {compass(ev.angle)}</span>
              <span className={"src "+ev.source.toLowerCase()}>{ev.source}</span>
            </div>
            <div className="pat">{patternFor(ev.urgency)}</div>
            <div className="sub">{(ev.distance*3.28084).toFixed(0)}ft · {ev.confidence}% · {ev.label.toLowerCase()}</div>
            <div className="det">
              <div className="kv"><span>id</span><span>{ev.id}</span></div>
              <div className="kv"><span>velocity</span><span>{ev.velocity>=0?"+":""}{(ev.velocity*3.28084).toFixed(1)} ft/s</span></div>
              <div className="kv"><span>eta</span><span>{ev.eta.toFixed(1)}s</span></div>
              <div className="kv"><span>audio</span><span>{ev.audioClass}</span></div>
              <div className="kv"><span>visual</span><span>{ev.visualClass}</span></div>
              <div className="summary">{summarize(ev)}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="log-foot">
        <span>{events.filter(e=>e.urgency==="critical").length} CRIT · {events.filter(e=>e.urgency==="high").length} HIGH</span>
        <button onClick={onAck}>ACK ALL</button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Info row — closest threat detail strip
// ─────────────────────────────────────────────
function InfoRow({ threats, heading }) {
  const closest = threats.length ? [...threats].sort((a,b)=>a.distance-b.distance)[0] : null;
  const critCount = threats.filter(t=>t.urgency==="critical").length;
  return (
    <div className="cell area-info">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>Nearest Threat</div>
        <div className="meta">
          <span>{threats.length} ACTIVE</span>
          <span>·</span>
          <span className={critCount?"live":""} style={critCount?{color:"#E24B4A"}:undefined}>
            {critCount?"⚠ CRITICAL":"NOMINAL"}
          </span>
        </div>
      </div>
      <div className="info">
        <div className="col">
          <div className="k">Subject</div>
          <div className="v">{closest ? closest.icon+" "+closest.label : "—"}</div>
          <div className="ind">{closest ? `${closest.source} · ${closest.audioClass}` : "no targets in range"}</div>
        </div>
        <div className="col">
          <div className="k">Bearing / Range</div>
          <div className="v">
            {closest ? `${compass(closest.angle)}` : "—"}
            <span className="u">{closest ? `${(closest.distance*3.28084).toFixed(0)} ft` : ""}</span>
          </div>
          <div className={"ind "+(closest && closest.urgency==="critical"?"red":"teal")}>
            {closest ? `${Math.round(closest.angle)}° abs · ${Math.round(((closest.angle-heading+540)%360)-180)}° rel` : ""}
          </div>
        </div>
        <div className="col">
          <div className="k">Closure / Conf</div>
          <div className="v">
            {closest ? `${closest.velocity>=0?"+":""}${(closest.velocity*3.28084).toFixed(1)}` : "—"}
            <span className="u">ft/s</span>
          </div>
          <div className="bars">
            {Array.from({length:10}).map((_,i)=>(
              <span key={i} className={closest && i < Math.round(closest.confidence/10) ? "on":""} style={{height: (4+i*1.4)+"px"}}/>
            ))}
          </div>
          <div className="ind">{closest ? `${closest.confidence}% conf · ETA ${closest.eta.toFixed(1)}s` : ""}</div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Status bar (bottom)
// ─────────────────────────────────────────────
function StatusBar({ stat, beams, signal, muted }) {
  const gemmaCls = stat.gemma<3?"green":stat.gemma<5?"yellow":"red";
  const pipeCls = stat.pipeline>8?"green":stat.pipeline>6?"yellow":"red";
  const yoloCls = stat.yolo<100?"green":stat.yolo<150?"yellow":"red";
  const tempCls = stat.temp<50?"green":stat.temp<60?"yellow":"red";
  const cpuCls = (stat.cpu_pct||0)<60?"green":(stat.cpu_pct||0)<80?"yellow":"red";
  const sigCls = signal===4?"green":signal===3?"yellow":signal===2?"yellow":"red";

  // mic bar heights (24px max)
  const mics = [beams.front,beams.right,beams.back,beams.left];

  return (
    <div className="statusbar">
      <div className="sb-item" title="Wearable identity">
        <span className="lbl">UNIT</span>
        <span className="val teal">WS-04F</span>
      </div>
      <div className="sb-item">
        <span className="lbl">GEMMA</span>
        <span className={"val "+gemmaCls}>{stat.gemma.toFixed(1)}s</span>
      </div>
      <div className="sb-item">
        <span className="lbl">PIPELINE</span>
        <span className={"val "+pipeCls}>{stat.pipeline.toFixed(1)}fps</span>
      </div>
      <div className="sb-item">
        <span className="lbl">YOLO</span>
        <span className={"val "+yoloCls}>{stat.yolo}ms</span>
      </div>
      <div className="sb-item">
        <span className="lbl">MICS</span>
        <span className="mics">
          {mics.map((e,i)=>(<span key={i} className={e>0.2?"on":""} style={{height: (4+e*20).toFixed(0)+"px"}}/>))}
        </span>
      </div>
      <div className="sb-item">
        <span className="lbl">TEMP</span>
        <span className={"val "+tempCls}>{stat.temp}°C</span>
      </div>
      <div className="sb-item">
        <span className="lbl">CPU</span>
        <span className={"val "+cpuCls}>{stat.cpu_pct||0}%</span>
      </div>
      <div className="sb-item">
        <span className="lbl">HAPTIC</span>
        <span className={"val "+(muted?"red":"teal")}>{muted?"MUTED":"ARMED"}</span>
      </div>
      <div className="sb-item">
        <span className="lbl">SIGNAL</span>
        <span className="sig">
          {[1,2,3,4].map(i=>(
            <span key={i} className={"lvl "+(i<=signal?"on":"")}>◆</span>
          ))}
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Threat modal
// ─────────────────────────────────────────────
function ThreatModal({ threat, onClose, onAck, onTrack }) {
  if(!threat) return null;
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={e=>e.stopPropagation()}>
        <div className="modal-hd">
          <div className="ttl">▸ THREAT · {threat.label}</div>
          <div className="id">{threat.id} · {threat.source}</div>
        </div>
        <div className="modal-bd">
          <div className="mb">
            <div className="k">Distance</div>
            <div className="v">{threat.distance.toFixed(2)}<span className="u">m</span></div>
          </div>
          <div className="mb">
            <div className="k">Bearing</div>
            <div className="v">{compass(threat.angle)} <span className="u">{Math.round(threat.angle)}°</span></div>
          </div>
          <div className="mb">
            <div className="k">Closure</div>
            <div className="v">{threat.velocity>=0?"+":""}{threat.velocity.toFixed(2)}<span className="u">m/s</span></div>
          </div>
          <div className="mb">
            <div className="k">ETA</div>
            <div className="v">{threat.eta.toFixed(1)}<span className="u">s</span></div>
          </div>
          <div className="mb">
            <div className="k">Confidence</div>
            <div className="v">{threat.confidence}<span className="u">%</span></div>
          </div>
          <div className="mb">
            <div className="k">Urgency</div>
            <div className="v" style={{color:{critical:"#E24B4A",high:"#BA7517",medium:"#FFD54F",low:"#888"}[threat.urgency]}}>{threat.urgency.toUpperCase()}</div>
          </div>
          <div className="mb full">
            <div className="k">Multimodal summary</div>
            <div className="summary">
              <strong style={{color:"#cfcfcf"}}>AUDIO</strong> · {threat.audioClass}<br/>
              <strong style={{color:"#cfcfcf"}}>VISUAL</strong> · {threat.visualClass}<br/>
              <strong style={{color:"#cfcfcf"}}>FUSION</strong> · {summarize(threat)}
            </div>
          </div>
        </div>
        <div className="modal-ft">
          <button onClick={()=>{onTrack(threat); onClose();}}>TRACK</button>
          <button className="danger" onClick={()=>{onAck(threat); onClose();}}>MUTE / ACK</button>
          <button onClick={onClose}>DISMISS</button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Live Camera PiP — actual webcam feed (operator verification cam)
// ─────────────────────────────────────────────
function LiveCamPiP({ enabled, onClose, onRequestEnable }) {
  const videoRef = useRef(null);
  const [state, setState] = useState("idle"); // idle | requesting | live | denied | error | unsupported
  const [err, setErr] = useState("");
  const streamRef = useRef(null);
  const [nowTs, setNowTs] = useState("");

  useEffect(()=>{
    if(!enabled){
      // stop stream
      if(streamRef.current){
        streamRef.current.getTracks().forEach(t=>t.stop());
        streamRef.current = null;
      }
      if(videoRef.current) videoRef.current.srcObject = null;
      setState("idle");
      return;
    }
    let cancelled = false;
    async function start(){
      if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
        setState("unsupported");
        return;
      }
      setState("requesting");
      try{
        const s = await navigator.mediaDevices.getUserMedia({
          video: { width:{ideal:480}, height:{ideal:360}, facingMode:"user" },
          audio: false
        });
        if(cancelled){ s.getTracks().forEach(t=>t.stop()); return; }
        streamRef.current = s;
        if(videoRef.current){
          videoRef.current.srcObject = s;
          videoRef.current.play().catch(()=>{});
        }
        setState("live");
      } catch(e){
        if(cancelled) return;
        const name = (e && e.name) || "";
        if(name==="NotAllowedError" || name==="PermissionDeniedError"){
          setState("denied"); setErr("permission denied");
        } else if(name==="NotFoundError" || name==="DevicesNotFoundError"){
          setState("error"); setErr("no camera detected");
        } else {
          setState("error"); setErr((e && e.message) || "stream failed");
        }
      }
    }
    start();
    return ()=>{
      cancelled = true;
      if(streamRef.current){
        streamRef.current.getTracks().forEach(t=>t.stop());
        streamRef.current = null;
      }
    };
  },[enabled]);

  // local timestamp tick when live
  useEffect(()=>{
    if(state!=="live") return;
    const id = setInterval(()=>{
      const d = new Date();
      const p = n=>String(n).padStart(2,"0");
      setNowTs(`${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`);
    }, 500);
    return ()=>clearInterval(id);
  },[state]);

  if(!enabled) return null;

  return (
    <div className="livecam">
      <video ref={videoRef} playsInline muted autoPlay/>
      <div className="lc-reticle" />
      <button className="lc-close" onClick={onClose} title="Close live cam">×</button>
      <div className="lc-hd">
        <span className={"nm "+(state==="live"?"":"off")}>
          {state==="live" ? "LIVE" : state==="requesting" ? "INIT" : "OFFLINE"}
        </span>
        <span style={{color:"var(--tx-2)"}}>OP-CAM</span>
      </div>
      {state==="live" && (
        <div className="lc-ft">
          <span>{nowTs}</span>
          <span>{streamRef.current?.getVideoTracks?.()[0]?.getSettings?.()?.width || ""}×{streamRef.current?.getVideoTracks?.()[0]?.getSettings?.()?.height || ""}</span>
        </div>
      )}
      {(state==="requesting" || state==="idle") && (
        <div className="lc-state">
          <div className="icon">⟳</div>
          <div>REQUESTING<br/>OPERATOR CAM…</div>
        </div>
      )}
      {state==="denied" && (
        <div className="lc-state err">
          <div className="icon">⊘</div>
          <div>ACCESS DENIED<br/><span style={{color:"var(--tx-3)"}}>{err}</span></div>
          <button onClick={onRequestEnable}>RETRY</button>
        </div>
      )}
      {state==="error" && (
        <div className="lc-state err">
          <div className="icon">!</div>
          <div>CAM FAULT<br/><span style={{color:"var(--tx-3)"}}>{err}</span></div>
          <button onClick={onRequestEnable}>RETRY</button>
        </div>
      )}
      {state==="unsupported" && (
        <div className="lc-state err">
          <div className="icon">⊘</div>
          <div>WEBRTC<br/>UNAVAILABLE</div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { CameraFeed, AudioBeams, HapticLog, InfoRow, StatusBar, ThreatModal, LiveCamPiP });
