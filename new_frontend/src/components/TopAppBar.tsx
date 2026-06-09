import { 
  Menu, 
  Search, 
  Bell, 
  Settings, 
  Sparkles, 
  Music 
} from "lucide-react";

interface TopAppBarProps {
  isSidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  currentView: string;
}

export default function TopAppBar({ 
  onToggleSidebar, 
  searchQuery, 
  setSearchQuery,
  currentView
}: TopAppBarProps) {
  return (
    <header className="fixed top-0 left-0 right-0 h-16 bg-surface/80 backdrop-blur-lg flex justify-between items-center px-6 md:px-10 z-50 border-b border-surface-variant/20">
      
      {/* Left side Brand logo */}
      <div className="flex items-center gap-4">
        <button 
          onClick={onToggleSidebar}
          className="p-2 rounded-lg hover:bg-surface-variant/20 text-on-surface-variant transition-all duration-300 active:scale-95"
          aria-label="Toggle sidebar menu"
        >
          <Menu className="w-5 h-5 text-on-surface" />
        </button>
        <div className="flex items-center gap-2 cursor-pointer">
          <div className="text-xl md:text-2xl font-display font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-secondary to-primary tracking-tight">
            VocalStars
          </div>
          <span className="hidden sm:inline-block bg-primary/15 text-primary text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
            AI Coach
          </span>
        </div>
      </div>

      {/* Center path or search indicator */}
      <div className="hidden md:flex items-center gap-6">
        {currentView === "dashboard" ? (
          <div className="relative flex items-center bg-surface-container-high rounded-full px-4 py-1.5 border border-white/5 w-80 lg:w-96 transition-all duration-300 focus-within:ring-2 focus-within:ring-primary/50 focus-within:shadow-[0_0_15px_rgba(255,177,192,0.2)]">
            <Search className="w-4 h-4 text-on-surface-variant mr-2" />
            <input 
              type="text" 
              placeholder="Search artists, songs or genres..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-transparent border-none outline-none focus:ring-0 text-sm w-full placeholder:text-on-surface-variant/50 text-on-surface"
            />
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
            <Music className="w-3.5 h-3.5 text-primary" />
            <span>Practice</span>
            <span className="text-on-surface-variant/40">/</span>
            <span className="text-primary font-bold">Vocal Studio</span>
          </div>
        )}
      </div>

      {/* Right control utilities */}
      <div className="flex items-center gap-4">
        <button 
          className="p-2 text-on-surface-variant hover:text-tertiary transition-all duration-350 hover:scale-110 relative"
          aria-label="Notifications"
        >
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full animate-pulse" />
        </button>
        
        <button 
          className="p-2 text-on-surface-variant hover:text-tertiary transition-all duration-350 hover:rotate-45"
          aria-label="Settings"
        >
          <Settings className="w-5 h-5" />
        </button>

        {/* Profile indicator */}
        <div className="flex items-center gap-3 pl-2 border-l border-white/10">
          <div className="w-8 h-8 rounded-full border-2 border-primary/30 overflow-hidden cursor-pointer hover:border-primary transition-all duration-300">
            <img 
              alt="User Profile" 
              className="w-full h-full object-cover" 
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80"
              referrerPolicy="no-referrer"
            />
          </div>
        </div>
      </div>
    </header>
  );
}
