"""
Railway Sales Performance Streamlit App (OOP, beginner-friendly)

Features:
- Load a CSV (file uploader or default path /mnt/data/railway.csv if present)
- Preprocessing (datetime parsing, fillna, drop duplicates)
- Peak vs Off-Peak analysis
- Top routes (overall / peak / off-peak)
- Ticket type & price visualizations
- Refund analysis & delays

Edit: Change class methods or add new visualizer functions. Each section is small and independent.
"""

import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="National Rail | Sales Performance", layout="wide")


# ---------- Data loader & preprocessor ----------
class RailDataLoader:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.df = None

    def load(self, uploaded_file) -> pd.DataFrame:
        """Load dataframe from uploaded_file (Streamlit) or fallback to self.path."""
        if uploaded_file is not None:
            self.df = pd.read_csv(uploaded_file)
            #st.success("Loaded file from upload.").T
        elif self.path and os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
            st.success(f"Loaded file from {self.path}")
        else:
            st.warning("No file provided and default file not found. Generating sample dataset.")
            self.df = self._generate_sample()
        return self.df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Do minimal, safe preprocessing. Defensive checks for column presence."""
        df = df.copy()
        # drop Transaction ID if exists
        if "Transaction ID" in df.columns:
            df.drop("Transaction ID", axis=1, inplace=True)

        # parse dates if present
        if "Date of Purchase" in df.columns:
            df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"], errors="coerce")

        # fill some common columns if present
        if "Railcard" in df.columns:
            df["Railcard"] = df["Railcard"].fillna(df["Railcard"].mode().iloc[0] if not df["Railcard"].mode().empty else "None")

        if "Reason for Delay" in df.columns:
            df["Reason for Delay"] = df["Reason for Delay"].fillna(df["Reason for Delay"].mode().iloc[0] if not df["Reason for Delay"].mode().empty else "Unknown")

        # If Arrival/Actual Arrival missing fill with Arrival Time
        if "Actual Arrival Time" in df.columns and "Arrival Time" in df.columns:
            df["Actual Arrival Time"] = df["Actual Arrival Time"].fillna(df["Arrival Time"])

        # remove duplicates
        df.drop_duplicates(inplace=True)

        # create combined departure datetime and hour if possible
        if "Date of Journey" in df.columns and "Departure Time" in df.columns:
            df["Departure DateTime"] = pd.to_datetime(df["Date of Journey"].astype(str) + " " + df["Departure Time"].astype(str), errors="coerce")
            df["Departure Hour"] = df["Departure DateTime"].dt.hour.fillna(-1).astype(int)
        else:
            # if missing create fake hour column to avoid breaking code; set to -1
            df["Departure Hour"] = -1

        # Create combined route label if both columns exist
        if "Departure Station" in df.columns and "Arrival Destination" in df.columns:
            df["Full Journey"] = df["Departure Station"].astype(str) + " to " + df["Arrival Destination"].astype(str)
        else:
            df["Full Journey"] = "Unknown route"

        # Ensure some expected columns exist for plotting; create defaults if not present
        for col, default in {
            "Ticket Type": "Unknown",
            "Price": 0.0,
            "Railcard": "None",
            "Refund Request": "No",
            "Journey Status": "On Time",
            "Payment Method": "Unknown",
            "Reason for Delay": "Unknown",
        }.items():
            if col not in df.columns:
                df[col] = default

        # ensure price numeric
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

        return df

    def _generate_sample(self, n=400) -> pd.DataFrame:
        """Generate a small sample dataset with expected columns for development/testing."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=30)
        stations = ["London Kings Cross", "York", "Liverpool Lime Street", "Manchester Piccadilly", "Bristol Temple Meads"]
        ticket_types = ["Advance", "Anytime", "Off-Peak"]
        journey_status = ["On Time", "Delayed", "Cancelled"]
        payment_methods = ["Card", "Cash", "Mobile"]
        railcards = ["None", "16-25 Railcard", "Senior Railcard"]

        data = {
            "Date of Purchase": rng.choice(dates, n),
            "Date of Journey": rng.choice(dates, n),
            "Departure Time": rng.integers(0, 23, n).astype(str) + ":00",
            "Departure Station": rng.choice(stations, n),
            "Arrival Destination": rng.choice(stations, n),
            "Ticket Type": rng.choice(ticket_types, n),
            "Price": np.round(rng.normal(50, 15, n), 2).clip(5),
            "Railcard": rng.choice(railcards, n),
            "Refund Request": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
            "Journey Status": rng.choice(journey_status, n, p=[0.8, 0.15, 0.05]),
            "Payment Method": rng.choice(payment_methods, n),
            "Reason for Delay": rng.choice(["Signal failure", "Weather", "Rolling stock", "Staffing"], n),
        }
        df = pd.DataFrame(data)
        # add Departure Hour computed from Departure Time
        df["Departure DateTime"] = pd.to_datetime(df["Date of Journey"].astype(str) + " " + df["Departure Time"], errors="coerce")
        df["Departure Hour"] = df["Departure DateTime"].dt.hour.fillna(-1).astype(int)
        df["Full Journey"] = df["Departure Station"] + " to " + df["Arrival Destination"]
        return df


# ---------- Analyzer ----------
class RailAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # define peak hours (edit these ranges if needed)
        self.peak_morning = list(range(6, 9))    # 6-8
        self.peak_evening = list(range(16, 19))  # 16-18
        self.off_peak = [h for h in range(24) if h not in self.peak_morning + self.peak_evening]

    def busiest_lines(self, top_n=10) -> pd.DataFrame:
        grouping = self.df.groupby(["Departure Station", "Arrival Destination"]).size().reset_index(name="Number of Journeys")
        grouping["Full Journey"] = grouping["Departure Station"].astype(str) + " to " + grouping["Arrival Destination"].astype(str)
        return grouping.sort_values("Number of Journeys", ascending=False).reset_index(drop=True).head(top_n)

    def top_routes_by_period(self, period="all", top_n=10) -> pd.DataFrame:
        if period == "peak":
            df_period = self.df[self.df["Departure Hour"].isin(self.peak_morning + self.peak_evening)]
        elif period == "off-peak":
            df_period = self.df[self.df["Departure Hour"].isin(self.off_peak)]
        else:
            df_period = self.df

        grouping = df_period.groupby(["Departure Station", "Arrival Destination"]).size().reset_index(name="Number of Journeys")
        grouping["Full Journey"] = grouping["Departure Station"].astype(str) + " to " + grouping["Arrival Destination"].astype(str)
        return grouping.sort_values("Number of Journeys", ascending=False).reset_index(drop=True).head(top_n)

    def departure_hour_counts(self) -> pd.Series:
        return self.df["Departure Hour"].value_counts().sort_index()

    def refund_stats(self, df_subset: Optional[pd.DataFrame] = None) -> pd.Series:
        if df_subset is None:
            df_subset = self.df
        return df_subset["Refund Request"].value_counts(normalize=True) * 100

    def refunds_by_delay_group(self, bins=None) -> pd.DataFrame:
        # if there's a delayed minutes column, use it; otherwise try to use 'Reason for Delay' as categorical grouping
        if "Delayed Minutes" in self.df.columns:
            b = bins or [0, 1, 5, 15, 30, 60, np.inf]
            self.df["DelayGroup"] = pd.cut(self.df["Delayed Minutes"], bins=b, right=False)
            out = self.df.groupby("DelayGroup").agg(
                DelayedMinutes=("Delayed Minutes", "count"),
                NetRevenue=("Price", "sum"),
                Refund=("Refund Request", lambda s: (s == "Yes").sum())
            ).reset_index()
            out["Refund%"] = np.round(out["Refund"] / out["DelayedMinutes"] * 100, 2)
            return out
        else:
            return pd.DataFrame()  # Nothing to compute

    # ----------------- NEW: top-by-revenue -----------------
    def top_routes_by_revenue(self, period="all", top_n=10):
        if period == "peak":
            df_period = self.df[self.df["Departure Hour"].isin(self.peak_morning + self.peak_evening)]
        elif period == "off-peak":
            df_period = self.df[self.df["Departure Hour"].isin(self.off_peak)]
        else:
            df_period = self.df

        if "Price" not in df_period.columns:
            return pd.DataFrame()

        grp = (df_period
               .groupby(["Departure Station", "Arrival Destination"], dropna=False)
               .agg(NumberOfJourneys=("Price", "size"),
                    Revenue=("Price", "sum"),
                    AvgPrice=("Price", "mean"))
               .reset_index())
        grp["Full Journey"] = grp["Departure Station"].astype(str) + " to " + grp["Arrival Destination"].astype(str)
        return grp.sort_values("Revenue", ascending=False).head(top_n)

    # ----------------- NEW: auto peak detection -----------------
    def detect_peak_hours_auto(self, coverage_pct=0.5):
        counts = self.df["Departure Hour"].value_counts().sort_values(ascending=False)
        if counts.empty:
            return []
        total = counts.sum()
        cum = counts.cumsum() / total
        hours = cum[cum <= coverage_pct].index.tolist()
        if not hours:
            hours = [counts.index[0]]
        return sorted([int(h) for h in hours])

    # ----------------- NEW: on-time performance -----------------
    def ontime_performance(self, by="overall", top_n=10):
        """
        by: 'overall', 'route', 'departure_station', 'arrival_destination'
        Returns a DataFrame with on-time rate and counts
        """
        if "Journey Status" not in self.df.columns:
            return pd.DataFrame()

        df = self.df.copy()
        df["OnTimeFlag"] = df["Journey Status"].apply(lambda s: 1 if str(s).lower().strip() == "on time" else 0)

        if by == "overall":
            total = len(df)
            ontime = df["OnTimeFlag"].sum()
            return pd.DataFrame([{"TotalJourneys": total, "OnTime": int(ontime), "OnTimePct": round(ontime/total*100 if total else 0,2)}])
        elif by == "route":
            grp = df.groupby(["Departure Station", "Arrival Destination"], dropna=False).agg(
                Total=("OnTimeFlag", "size"),
                OnTime=("OnTimeFlag", "sum"))
            grp["OnTimePct"] = (grp["OnTime"] / grp["Total"] * 100).round(2)
            grp = grp.reset_index()
            grp["Full Journey"] = grp["Departure Station"].astype(str) + " to " + grp["Arrival Destination"].astype(str)
            return grp.sort_values("OnTimePct", ascending=False).head(top_n).reset_index(drop=True)
        elif by == "departure_station":
            grp = df.groupby("Departure Station", dropna=False).agg(Total=("OnTimeFlag","size"), OnTime=("OnTimeFlag","sum"))
            grp["OnTimePct"] = (grp["OnTime"]/grp["Total"]*100).round(2)
            return grp.reset_index().sort_values("OnTimePct", ascending=False).head(top_n).reset_index(drop=True)
        elif by == "arrival_destination":
            grp = df.groupby("Arrival Destination", dropna=False).agg(Total=("OnTimeFlag","size"), OnTime=("OnTimeFlag","sum"))
            grp["OnTimePct"] = (grp["OnTime"]/grp["Total"]*100).round(2)
            return grp.reset_index().sort_values("OnTimePct", ascending=False).head(top_n).reset_index(drop=True)
        else:
            return pd.DataFrame()


# ---------- Visualizer ----------
class RailVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")

    def plot_top_routes_bar(self, df_routes: pd.DataFrame, title="Top Routes - Number of Journeys"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Number of Journeys", y="Full Journey", data=df_routes, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Number of Journeys")
        ax.set_ylabel("")
        plt.tight_layout()
        return fig

    def plot_departure_hour(self, hour_counts: pd.Series, title="Number of Journeys by Departure Hour"):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Journeys")
        plt.tight_layout()
        return fig

    def plot_ticket_type_counts(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Ticket Type", data=df, ax=ax, order=df["Ticket Type"].value_counts().index)
        ax.set_title("Ticket Type Counts")
        plt.tight_layout()
        return fig

    def plot_ticket_price_by_type(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Ticket Type", y="Price", hue="Railcard", data=df, ax=ax)
        ax.set_title("Ticket Type vs Price (by Railcard)")
        plt.tight_layout()
        return fig

    def plot_refund_price(self, df: pd.DataFrame, title="Price Distribution for Refund vs No Refund"):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Refund Request", y="Price", hue="Journey Status", data=df, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def plot_reason_delay(self, df: pd.DataFrame):
        # Countplot for Reason for Delay vs Journey Status
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x="Reason for Delay", hue="Journey Status", data=df, ax=ax, order=df["Reason for Delay"].value_counts().index)
        ax.set_title("Journey Status by Reason for Delay")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig
    #---------------------------------------------------------------new for top ontime line charts
    def plot_ontime_routes_bar(self, df_routes, title="Top Routes by On-Time % (Top 10)"):
        """
        Expects df_routes to have columns: 'Full Journey' and 'OnTimePct'.
        Produces a horizontal bar chart sorted by OnTimePct (highest at top).
        """
        # defensive copy & sort
        df = df_routes.copy()
        if "OnTimePct" not in df.columns or "Full Journey" not in df.columns:
            # fallback: return empty fig with message
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "OnTimePct or Full Journey column missing", ha="center", va="center")
            ax.axis("off")
            return fig

        # sort so highest on-time % appears at top
        df_sorted = df.sort_values("OnTimePct", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        order = df_sorted["Full Journey"].tolist()
        sns.barplot(x="OnTimePct", y="Full Journey", data=df_sorted, order=order, ax=ax, orient="h")
        ax.set_xlim(0, 100)
        ax.set_xlabel("On-Time Percentage (%)")
        ax.set_ylabel("")
        ax.set_title(title)

        # annotate bars with percent values
        for p in ax.patches:
            width = p.get_width()
            # position at end of bar; +1 shift to separate from bar
            ax.text(width + 1, p.get_y() + p.get_height() / 2,
                    f"{width:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        return fig
 
    # ----------------- NEW: top routes by revenue -----------------
    def plot_top_routes_revenue(self, df_routes, title="Top Routes by Revenue"):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x="Revenue", y="Full Journey", data=df_routes, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Revenue (sum)")
        ax.set_ylabel("")
        plt.tight_layout()
        return fig

    # ----------------- NEW: heatmap day vs hour -----------------
    def plot_heatmap_day_hour(self, df, date_col="Date of Journey", hour_col="Departure Hour"):
        if date_col not in df.columns or hour_col not in df.columns:
            return None
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2["dow"] = df2[date_col].dt.day_name().fillna("Unknown")
        pivot = pd.crosstab(df2["dow"], df2[hour_col])
        # reorder days
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = pivot.reindex([d for d in days if d in pivot.index])
        fig, ax = plt.subplots(figsize=(12,4))
        sns.heatmap(pivot.fillna(0), annot=False, ax=ax)
        ax.set_title("Journeys by Day of Week vs Hour")
        plt.tight_layout()
        return fig


# ---------- Streamlit App ----------
class RailDashboardApp:
    def __init__(self):
        self.loader = RailDataLoader(path="/railway.csv")  # default path (edit as needed)
        self.visualizer = RailVisualizer()
        self.analyzer = None
        self.df = None

    def run(self):
        st.title("NATIONAL RAIL | Sales Performance")
        #st.caption("Interactive dashboard — organized with OOP (easy to edit)")

        # Sidebar: file upload and options
        st.sidebar.header("Data & Controls")
        uploaded_file = st.sidebar.file_uploader("Upload railway CSV", type=["csv"])
        df_loaded = self.loader.load(uploaded_file)
        self.df = self.loader.preprocess(df_loaded)
        self.analyzer = RailAnalyzer(self.df)

        # Basic Info
        st.sidebar.subheader("Dataset Info")
        st.sidebar.write(f"Rows: {self.df.shape[0]}")
        st.sidebar.write(f"Columns: {self.df.shape[1]}")
        top_n = st.sidebar.slider("Top N routes to show", 5, 20, value=10)

        # Filters
        st.sidebar.subheader("Time / Period Filters")
        period = st.sidebar.selectbox("Period", options=["all", "peak", "off-peak"], index=0)
        show_table = st.sidebar.checkbox("Show raw data preview", value=False)
###############################################################################################################
        # Main layout - create columns similar to dashboard
        # Row 1: Top routes and monthly / trend boxes (simplified)
        tab1, tab2, tab3, tab4 = st.tabs(['Most Popular Routes', 'peak travel times', 'Sales Performance', 'on-time performance and contributing factors'])

        with tab1:
            col1, col2 = st.columns([2, 3])
                
            with col1:
                st.subheader("Top Routes")
                top_routes = self.analyzer.top_routes_by_period(period=period, top_n=top_n)
                if not top_routes.empty:
                    fig = self.visualizer.plot_top_routes_bar(top_routes, title=f"Top {top_n} Routes ({period})")
                    st.pyplot(fig)
                    st.dataframe(top_routes[["Full Journey", "Number of Journeys"]].head(top_n).reset_index(drop=True))
                else:
                    st.info("Not enough data to compute top routes for the selected period.")

                # NEW: optionally show top-by-revenue for the same period
                # with st.expander("Show Top Routes by Revenue"):
                #     top_rev = self.analyzer.top_routes_by_revenue(period=period, top_n=top_n)
                #     if not top_rev.empty:
                #         st.pyplot(self.visualizer.plot_top_routes_revenue(top_rev))
                #         st.dataframe(top_rev[["Full Journey", "NumberOfJourneys", "Revenue", "AvgPrice"]].round(2).reset_index(drop=True))
                #     else:
                #         st.info("No revenue data available to compute top-by-revenue.")

            with col2:
                top_rev = self.analyzer.top_routes_by_revenue(period=period, top_n=top_n)
                if not top_rev.empty:
                    st.pyplot(self.visualizer.plot_top_routes_revenue(top_rev))
                    st.dataframe(top_rev[["Full Journey", "NumberOfJourneys", "Revenue", "AvgPrice"]].round(2).reset_index(drop=True))
                else:
                    st.info("No revenue data available to compute top-by-revenue.")

            # Row 2: Ticket distribution and price by railcard
        with tab2:
            right,left=st.columns([1,1])
            # Auto-detected peak hours
            auto_peak = self.analyzer.detect_peak_hours_auto(coverage_pct=0.5)
            
            with left:
                

                st.subheader("Heatmap OF Peak Hour Daily")
                # show heatmap day vs hour
                heat = self.visualizer.plot_heatmap_day_hour(self.df)
                if heat:
                    st.pyplot(heat)
                else:
                    st.info("Not enough date/hour columns to show heatmap. Ensure 'Date of Journey' and 'Departure Hour' exist.")
                
                st.markdown('---')
                st.write(f"Auto-detected hours that cover ~50% of journeys: {auto_peak}")
                st.write(f"Defined peak morning: {self.analyzer.peak_morning}, peak evening: {self.analyzer.peak_evening}")
            with right:
                st.subheader("Departure Hour Distribution")
                hour_counts = self.analyzer.departure_hour_counts()
                if len(hour_counts) == 0:
                    st.info("No departure hour data available.")
                else:
                    fig2 = self.visualizer.plot_departure_hour(hour_counts)
                    st.pyplot(fig2)
                        
                # list top hours table
                st.subheader("Top Hours by Number of Journeys")
                top_hours = self.df["Departure Hour"].value_counts().reset_index()
                top_hours.columns = ["Hour", "Journeys"]
                st.dataframe(top_hours.sort_values("Journeys", ascending=False).head(24).reset_index(drop=True))

        with tab3:
            col3, col4 = st.columns(2)
            with col3:
                ######################################################################################Ticket Type Distribution
                # st.subheader("Ticket Type Distribution")
                # fig3 = self.visualizer.plot_ticket_type_counts(self.df)
                # st.pyplot(fig3)
                st.subheader("Ticket Type vs Price (by Railcard)")
                fig4 = self.visualizer.plot_ticket_price_by_type(self.df)
                st.pyplot(fig4)


                # NEW: Top routes by revenue quick view
                st.subheader("Top Routes by Revenue (quick)")
                top_rev_small = self.analyzer.top_routes_by_revenue(period=period, top_n=5)
                if not top_rev_small.empty:
                    st.table(top_rev_small[["Full Journey","NumberOfJourneys","Revenue"]].round(2).reset_index(drop=True))
                else:
                    st.info("No revenue route data available.")

            with col4:
                

                # NEW: Revenue by ticket type
                #st.markdown("---")
                st.subheader("Revenue by Ticket Type")
                rev_by_type = (self.df.groupby("Ticket Type", dropna=False)
                                .agg(TotalRevenue=("Price","sum"),
                                    Count=("Price","size"),
                                    AvgPrice=("Price","mean"))
                                .reset_index()
                                .sort_values("TotalRevenue", ascending=False))
                st.dataframe(rev_by_type.round(2))

                fig_revtype, ax = plt.subplots(figsize=(8,3))
                sns.barplot(x="Ticket Type", y="TotalRevenue", data=rev_by_type, ax=ax)
                ax.set_title("Revenue by Ticket Type")
                st.pyplot(fig_revtype)

        with tab4:
            # Row 3: Refund analysis / delays
            #st.markdown("---")
            #st.subheader("Refunds & Delay Analysis")
            left, right = st.columns([2, 3])
            with left:
                subleftleft, subleftright=st.columns([1,1])
                with subleftleft:
                        
                    st.write("Refund percentages (overall):")
                    refunds_overall = self.analyzer.refund_stats()
                    st.write(refunds_overall.round(2))
                with subleftright:
                        
                    st.write(f"Refund percentages ({period}):")
                    if period == "peak":
                        subset = self.df[self.df["Departure Hour"].isin(self.analyzer.peak_morning + self.analyzer.peak_evening)]
                    elif period == "off-peak":
                        subset = self.df[self.df["Departure Hour"].isin(self.analyzer.off_peak)]
                    else:
                        subset = self.df
                    st.write(self.analyzer.refund_stats(subset).round(2))

                # Refund price distribution
                st.write("Price distribution by refund request:")
                fig_refund = self.visualizer.plot_refund_price(subset, title=f"Refund vs Price ({period})")
                st.pyplot(fig_refund)

                # # If Delayed Minutes exists show grouped refund stats
                # st.markdown("### Refunds by Delay Group (if `Delayed Minutes` exists)")
                # delay_group_table = self.analyzer.refunds_by_delay_group()
                # if not delay_group_table.empty:
                #     st.dataframe(delay_group_table)
                # else:
                #     st.info("No `Delayed Minutes` column found or insufficient data for delay-group analysis.")
                
                #####################----------------------------------------------------------------
                st.write("Reason for Delay vs Journey Status")
                fig_delay = self.visualizer.plot_reason_delay(self.df)
                st.pyplot(fig_delay)

                # NEW: On-time performance summary
                st.markdown("### On-time Performance")
                overall_ontime = self.analyzer.ontime_performance(by="overall")
                if not overall_ontime.empty:
                    #st.write("Overall on-time KPI:")
                    st.write(overall_ontime.T)
                else:
                    st.info("No 'Journey Status' column to compute on-time KPI.")

            with right:
                #top on time Full journey
                ontime_routes = self.analyzer.ontime_performance(by="route", top_n=10)
                if not ontime_routes.empty:
                    st.dataframe(ontime_routes[["Full Journey","Total","OnTime","OnTimePct"]])
                else:
                    st.info("Not enough data to compute per-route on-time performance.")
                #Repeated and not improtant 
                # ontime_routes = self.analyzer.ontime_performance(by="route", top_n=10)
                # if not ontime_routes.empty:
                #     # show chart first
                #     fig_ontime = self.visualizer.plot_ontime_routes_bar(ontime_routes)
                #     st.pyplot(fig_ontime)

                #     # then show the numeric table
                #     st.dataframe(ontime_routes[["Full Journey","Total","OnTime","OnTimePct"]].reset_index(drop=True))
                # else:
                #     st.info("Not enough data to compute per-route on-time performance.")
#############---------------------
                

                # # per-route on-time performance (top routes)
                # st.write("Top routes by on-time % (top 10):")
                # ontime_routes = self.analyzer.ontime_performance(by="route", top_n=10)
                # if not ontime_routes.empty:
                #     st.dataframe(ontime_routes[["Full Journey","Total","OnTime","OnTimePct"]])
                # else:
                #     st.info("Not enough data to compute per-route on-time performance.")

                # per-departure-station on-time
                st.write("Top departure stations by on-time % (top 10):")
                dep_ontime = self.analyzer.ontime_performance(by="departure_station", top_n=10)
                if not dep_ontime.empty:
                    st.dataframe(dep_ontime)
                else:
                    st.info("Not enough data to compute per-station on-time performance.")

            # Lower section: raw data and downloadable csv option
            st.markdown("---")
            if show_table:
                st.subheader("Raw Data Preview (first 200 rows)")
                st.dataframe(self.df.head(200))
        # offer download of processed CSV
        csv = self.df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download processed CSV", data=csv, file_name="rail_processed.csv", mime="text/csv")

        # Helpful tips for beginners
        st.markdown(
            """
            **How to edit this app (quick):**
            - Change peak hours in `RailAnalyzer.__init__`.
            - Add new plots: create a function in `RailVisualizer` and call it in `RailDashboardApp.run`.
            - To use a different default file, edit `RailDataLoader(path=...)`.
            """
        )


if __name__ == "__main__":
    app = RailDashboardApp()
    app.run()
