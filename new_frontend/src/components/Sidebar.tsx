import { motion } from "motion/react";
import {
  Bug,
  Mic,
  LayoutDashboard,
  Dumbbell,
  History,
  Radio,
} from "lucide-react";

interface SidebarProps {
  currentView: string;
  onNavigate: (view: string) => void;
  isCollapsed: boolean;
  setIsCollapsed: (collapsed: boolean) => void;
}

export default function Sidebar({ 
  currentView, 
  onNavigate, 
  isCollapsed, 
  setIsCollapsed 
}: SidebarProps) {
  
  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "studio", label: "Studio Session", icon: Mic },
    { id: "exercises", label: "Exercises", icon: Dumbbell },
    { id: "history", label: "History", icon: History },
    { id: "ai-debug", label: "AI Debug", icon: Bug },
  ];

  return (
    <aside 
      className={`fixed left-0 top-16 bottom-0 z-40 bg-surface-container border-r border-white/5 flex flex-col py-6 transition-all duration-300 overflow-hidden ${
        isCollapsed ? "w-20" : "w-64"
      }`}
      id="sidebar"
    >
      {/* Studio Header Status */}
      <div className={`px-4 mb-8 flex items-center transition-all ${isCollapsed ? "justify-center" : "gap-3 px-6"}`}>
        <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-secondary to-primary flex items-center justify-center glow-pink transition-transform duration-300 hover:scale-110">
          <Radio className="w-5 h-5 text-on-primary animate-pulse" />
        </div>
        {!isCollapsed && (
          <motion.div 
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex flex-col"
          >
            <h2 className="font-display font-bold text-sm text-on-surface whitespace-nowrap">Studio Session</h2>
            <p className="text-[10px] uppercase tracking-wider text-tertiary flex items-center gap-1">
              <span className="block w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse" />
              On Air
            </p>
          </motion.div>
        )}
      </div>

      {/* Navigation list */}
      <nav className="flex-1 px-3 space-y-2">
        {navItems.map((item) => {
          const isActive = currentView === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center rounded-lg transition-all duration-300 ${
                isCollapsed ? "justify-center p-3" : "gap-4 px-4 py-3"
              } ${
                isActive
                  ? "bg-gradient-to-r from-secondary/20 to-primary/20 text-primary border-r-4 border-primary"
                  : "text-on-surface-variant hover:bg-white/5 hover:text-primary"
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? "text-primary" : "text-on-surface-variant group-hover:text-primary"}`} />
              {!isCollapsed && (
                <span className="font-medium text-sm text-left">{item.label}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Bottom Button */}
      <div className={`mt-auto px-3 ${isCollapsed ? "flex justify-center" : ""}`}>
        <button
          onClick={() => onNavigate("studio")}
          className={`w-full py-3.5 rounded-xl bg-gradient-to-r from-secondary to-primary text-on-primary font-bold hover:brightness-110 hover:shadow-[0_0_20px_rgba(255,177,192,0.5)] active:scale-95 transition-all duration-300 shadow-lg glow-pink flex items-center justify-center gap-2`}
        >
          <Mic className="w-4 h-4" />
          {!isCollapsed && (
            <span className="text-xs tracking-wider uppercase">Live Studio</span>
          )}
        </button>
      </div>
    </aside>
  );
}
